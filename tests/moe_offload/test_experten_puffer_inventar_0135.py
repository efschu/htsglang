"""#135: der Experten-Puffer steht im Inventar -- also im Manifest und im Plan.

DER BEFUND, gemessen an fnFL2w67 (9a50ccbb34):

    MANIFEST-WRITE group=P rank=0 pieces=1103 bytes=3005657056   (3,0 GB)
    WEG2-XCHG-COVER  18 Chunk-Tags, zusammen 3,59 GiB

gegen 13/20/13 GB Kartenbelegung. Der Presplit ersetzt den Experten-
Parameter durch einen 0-Zeilen-Platzhalter und haelt die Bytes im
Slot-Puffer; `card_inventory` lief nur ueber `named_parameters()` und sah
davon NICHTS. Manifest, Cross-Group-Join und Plan erbten die Luecke.

WARUM EIN PUFFER UND NICHT JE BAND (w67 hat das widerlegt): `_nbytes` misst
den STORAGE, weil `covered_storage` per Storage-Key deckt -- ein View meldet
damit den ganzen Puffer (mib=800.000 fuer 25 MiB) und die Coverage refuest
(W84). Ein Band kann auch kein eigener Tensor sein, der MoE-Kernel braucht
den Stapel zusammenhaengend. Der Schnitt ueber die Experten gehoert deshalb
in die SHARD-ACHSE, die der Join aus den Manifesten abliest.
"""
import torch

import pytest

from sglang.srt.managers import weg2_memory_saver as ms
from sglang.srt.weg2 import weight_exchange_shadow as sh


@pytest.fixture
def chunks(monkeypatch):
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "16")
    monkeypatch.delenv(ms.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(ms.EXPERT_BAND_ENV_COUNT, raising=False)


def _modell(puffer=None, layer_id=7):
    experts = torch.nn.Module()
    experts.w13_weight_packed = torch.nn.Parameter(
        torch.zeros(0, 4), requires_grad=False)     # der 0-Zeilen-Platzhalter
    if puffer is not None:
        setattr(experts, ms.expert_buffer_attr_name("w13_weight_packed"), puffer)
    mlp = torch.nn.Module(); mlp.experts = experts
    lay = torch.nn.Module(); lay.mlp = mlp
    layers = torch.nn.ModuleList(
        [torch.nn.Module() for _ in range(layer_id)] + [lay])
    model = torch.nn.Module(); model.layers = layers
    root = torch.nn.Module(); root.model = model
    return root


# --- die Namens-Naht: eine Stelle schreibt, eine liest --------------------

def test_name_und_erkennung_gehoeren_zusammen():
    n = ms.expert_buffer_attr_name("w13_weight_packed")
    assert n == "weg2_experts_w13_weight_packed"
    assert ms.is_expert_buffer_attr(n)
    assert ms.is_expert_buffer_attr("model.layers.7.mlp.experts." + n)
    assert not ms.is_expert_buffer_attr("w13_weight_packed")
    assert not ms.is_expert_buffer_attr("model.layers.7.self_attn.qkv_proj.weight")


# --- ohne Puffer: unveraendert -------------------------------------------

def test_ohne_puffer_aendert_sich_nichts(chunks):
    inv, skipped, walked, reason = sh.card_inventory(rank=0, model=_modell())
    namen = [g.name for g, _t in (inv or [])]
    assert not [n for n in namen if "weg2_experts" in n]


# --- mit Puffer: er ist drin, mit den RICHTIGEN Bytes --------------------

def test_der_puffer_steht_im_inventar_mit_seinen_echten_bytes(chunks):
    # [220 Slots, 160, 2560] int32 -- die Form, die w67 am Metall hatte
    buf = torch.zeros(220, 16, 64, dtype=torch.int32)
    inv, skipped, walked, reason = sh.card_inventory(rank=0, model=_modell(buf))
    assert inv, reason
    treffer = [g for g, _t in inv if "weg2_experts" in g.name]
    assert len(treffer) == 1, [g.name for g, _t in inv]
    g = treffer[0]
    assert g.name == ("model.layers.7.mlp.experts."
                      "weg2_experts_w13_weight_packed")
    # Layer 7 -> Chunk 7//3 = 2. Der CHUNK-Tag, nicht ein Band-Tag: die
    # Feinheit kommt aus der Shard-Achse, nicht aus dem Namen.
    assert g.tag == "weights_2"
    # Die Bytes sind die des Tensors, nicht die eines Storages daneben:
    # StorageGeom flacht [220,16,64] auf rows=220*16, cols=64 ab.
    assert g.rows_full * g.cols_full * g.itemsize == buf.numel() * buf.element_size()
    assert g.rows_full == 220 * 16 and g.cols_full == 64


def test_der_puffer_reist_mit_seinem_tensor(chunks):
    buf = torch.zeros(8, 4, 16, dtype=torch.int32)
    inv, _s, _w, _r = sh.card_inventory(rank=0, model=_modell(buf))
    paar = [(g, t) for g, t in inv if "weg2_experts" in g.name]
    assert len(paar) == 1
    _g, t = paar[0]
    assert t.data_ptr() == buf.data_ptr(), (
        "Geometrie und Tensor muessen aus DEMSELBEN Walk-Schritt kommen -- "
        "sie nachtraeglich per Name zu paaren waere die zweite Buchhaltung"
    )


def test_ein_leerer_puffer_kommt_NICHT_ins_inventar(chunks):
    """0 Zeilen = kein Byte zu bewegen; im Inventar waere es ein Parameter
    ohne Deskriptor -> W74 Weg2XchgSourceMissing.

    ZWEI SICHERUNGEN, und der Mutantenlauf hat das gezeigt statt es zu
    behaupten: die `numel() == 0`-Pruefung in der Schleife ueberlebt ihre
    Entfernung, weil `ParamGeom.of` dieselbe Form ohnehin verweigert
    ("W68 ... extents (0, 16) itemsize 4 do not describe a tensor") und der
    Skip-Pfad sie als `undescribable-W68` auffaengt. Die frueh-Pruefung
    bleibt trotzdem: sie haelt eine erwartete Form aus der Skip-LISTE, in
    der sonst je Boot 48 Layer x 4 Attribute als Defekt gezaehlt wuerden.
    Dieser Test bindet das ERGEBNIS, und der naechste bindet die Herkunft.
    """
    buf = torch.zeros(0, 4, 16, dtype=torch.int32)
    inv, skipped, _w, _r = sh.card_inventory(rank=0, model=_modell(buf))
    assert not [g for g, _t in (inv or []) if "weg2_experts" in g.name]
    assert not [n for n, _why in skipped if "weg2_experts" in n], (
        "der leere Puffer landete in der Skip-Liste statt frueh auszusteigen "
        "-- dann zaehlt jeder Boot 48x4 erwartete Formen als Defekt"
    )


def test_die_bytes_des_manifests_stimmen_mit_dem_tensor(chunks):
    """Bis ins Manifest durchgerechnet: pieces_from_inventory nimmt
    rows*cols*itemsize AUS DER GEOMETRIE -- genau die Stelle, an der der
    View-Weg (Storage) falsch gerechnet haette."""
    from sglang.srt.weg2 import xchg_manifest as xm

    buf = torch.zeros(220, 16, 64, dtype=torch.int32)
    inv, _s, _w, _r = sh.card_inventory(rank=0, model=_modell(buf))
    stuecke = xm.pieces_from_inventory([g for g, _t in inv])
    meins = [p for p in stuecke if "weg2_experts" in p.param_name]
    assert len(meins) == 1
    assert meins[0].nbytes == buf.numel() * buf.element_size() == 220 * 16 * 64 * 4
    assert meins[0].tag == "weights_2"


# --- #136: BEIDE Inventare lesen dieselbe Quelle ---------------------------

def test_plan_und_manifest_sehen_DENSELBEN_puffer(chunks):
    """fnFL2w68: das Manifest trug die Experten (43,34 GB statt 5,19 GB),
    der PLAN nicht -- `derive_leg_plan` hatte eine EIGENE
    `named_parameters()`-Schleife. Die Coverage haelt den Plan gegen die
    lebenden Tensoren und meldete sie als UNCOVERED (W84: 116/32/44
    findings). Zwei Buchhaltungen, die an der teuersten Stelle auseinander-
    laufen -- genau die Klasse, vor der `card_inventory`s Docstring warnt.
    """
    buf = torch.zeros(220, 16, 64, dtype=torch.int32)
    model = _modell(buf)
    # Die EINE Quelle, die beide lesen:
    aus_der_quelle = dict(sh.expert_buffer_tensors(model))
    assert len(aus_der_quelle) == 1
    name = next(iter(aus_der_quelle))
    assert name.endswith("weg2_experts_w13_weight_packed")
    assert aus_der_quelle[name].data_ptr() == buf.data_ptr()

    inv, _s, _w, _r = sh.card_inventory(rank=0, model=model)
    im_inventar = {g.name for g, _t in inv}
    assert name in im_inventar, (
        "das Manifest-Inventar sieht den Puffer nicht"
    )


def test_der_helfer_ist_die_einzige_stelle(chunks):
    """Gegenprobe gegen die Rueckkehr der zweiten Buchhaltung: beide
    Inventare muessen `expert_buffer_tensors` RUFEN, nicht eine eigene
    Schleife ueber `vars(module)` fuehren."""
    import inspect

    for fn in (sh.card_inventory, sh.derive_leg_plan):
        # NUR CODE, keine Kommentare: die Kommentare NENNEN `vars(module)`
        # genau dort, wo sie erklaeren, warum der Helfer noetig ist -- ein
        # Test, der den Erklaertext trifft, misst die Prosa statt die Naht.
        src = "\n".join(z.split("#")[0] for z in inspect.getsource(fn).split("\n"))
        assert "expert_buffer_tensors(model)" in src, fn.__name__
        # Die eigentliche Aussage: KEIN eigener Walk ueber die Modul-Dicts.
        # (`is_expert_buffer_attr` darf sonst vorkommen -- #137 fragt damit
        # in der Paar-Verengung, ob ein Eintrag ein Experten-Puffer IST, und
        # das ist kein zweites Inventar, sondern eine Eigenschaftsfrage an
        # einen Eintrag, den der Helfer geliefert hat.)
        assert "vars(module)" not in src, (
            f"{fn.__name__} walkt die Modul-Dicts selbst statt den Helfer zu "
            f"rufen -- das ist die zweite Buchhaltung, die w68 gekostet hat"
        )


# --- #137: die Verengung auf das co-lokierte Paar -------------------------

def test_experten_puffer_ueberleben_die_paar_verengung():
    """fnFL2w69: `reconcile_card_manifest` verengt den Plan auf die
    Schnittmenge der Manifest-IDENTITAETEN des co-lokierten Paares, und eine
    Identitaet enthaelt `rows_full`. Ein Experten-Puffer hat je Gruppe eine
    andere (P-Stufe 0: 188+32 Slots, D-Rang 0: 134+Scratch), also treffen
    sich die Identitaeten nie und die Schnittmenge wirft ihn raus -- jeder
    Rang meldete gleichzeitig uncovered=12 UND missing=12.

    Dass sie ungleich sind, ist der GRUND, warum der Flip sie bewegt. Dieser
    Test bindet die Ausnahme an den Namen, nicht an eine Gesinnung: ein
    normaler Tensor MUSS weiter an der Geometrie scheitern.
    """
    import inspect

    src = inspect.getsource(sh.derive_leg_plan)
    i = src.index("kept = [g for g in inventory")
    block = src[i:i + 320]
    assert "is_expert_buffer_attr(g.name)" in block, (
        "die Verengung nimmt die Experten-Puffer nicht aus -- sie fallen "
        "wieder heraus, und der Flip transportiert erneut null Experten"
    )
    assert "manifest_entry(" in block, (
        "die Geometrie-Pruefung ist ganz entfallen -- sie MUSS fuer jeden "
        "anderen Tensor stehen bleiben, sonst faengt nichts mehr echte "
        "Divergenz (W80)"
    )
