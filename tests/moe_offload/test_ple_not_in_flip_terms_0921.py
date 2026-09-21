"""#78 (21.09., Nutzer): was nie geladen wird, gehoert in keinen Flip-Term.

Die Per-Layer-Embeddings von Next Flash werden zur LAUFZEIT per mmap
gelesen -- #54 hat ihren Gather mit 8,6 ms je Runde aus mmap-Shards
gemessen. Sie erreichen weder VRAM noch den Host-Store, also traegt sie
auch kein Flip. Der Checkpoint-Zensus zaehlte sie trotzdem mit, und daraus
sizet der Bounce:

    WEG2-XCHG-BOUNCE PUBLISHED ... widest_layer_bytes=103797370776,
    max_tag_bytes=106524281944      (fnFL2w4, 21.09. 17:53Z)
    WEG2-XCHG-TAGMAX lane=0 max_tag_mib=101589

GEMESSEN am Shipped-Checkpoint (163,20 GiB, 48 safetensors):

    widest layer   3-Layer-Band   Summe
    mit PLE     96,67 GiB   99,21 GiB   156,29 GiB
    ohne PLE     1,27 GiB    3,81 GiB    60,89 GiB

Faktor 26 auf dem Band, 76 auf dem breitesten Layer -- weil die PLE fast
vollstaendig an EINEM Layer haengen (Layer 1: 96,67 GiB gegen 1,27 GiB bei
Layer 0 und 2).

NICHT die Ursache des Container-OOM, und das gehoert danebengeschrieben,
damit niemand die falsche Lehre zieht: am laufenden Boot gemessen ist
`shmem` (der tmpfs-Experten-Store) mit 61,10 GiB der Posten, der echte
Datei-Cache nur ~4,4 GiB.
"""

from sglang.srt.weg2 import checkpoint_census as cc


def test_ple_is_excluded_by_segment_not_by_prefix():
    """`ple` steht MITTEN im Namen -- derselbe Layer traegt geladene und nie
    geladene Tensoren nebeneinander, anders als beim MTP-Baum."""
    assert cc.PLE_SEGMENTS == (".ple.",)
    nm = "model.language_model.layers.1.ple.key_proj.weight_packed"
    assert not nm.startswith(cc.MTP_TREE_PREFIXES), "kein Praefix-Fall"
    assert any(s in nm for s in cc.PLE_SEGMENTS)


def test_a_module_merely_containing_ple_is_not_swept_up():
    """Die Punkte sind der Schutz: `simple_proj` oder `ample.weight` duerfen
    nicht mitgenommen werden."""
    for nm in ("model.layers.0.sample.weight",
               "model.layers.0.ample_proj.weight",
               "model.layers.0.mlp.ple_gate.weight"):
        assert not any(s in nm for s in cc.PLE_SEGMENTS), nm


def test_the_exclusion_is_wired_into_the_widest_layer_reader():
    """Der Leser, der den Bounce sizet, muss die Liste FRAGEN -- sonst ist
    sie wieder ein Werkzeug ohne Verbraucher."""
    import inspect

    src = inspect.getsource(cc)
    assert "exclude_segments=PLE_SEGMENTS" in src, (
        "widest_layer_terms muss PLE_SEGMENTS durchreichen")


def test_the_census_honours_both_exclusions_independently():
    """Reine Funktionspruefung ohne Checkpoint: beide Filter greifen, und
    keiner verschluckt den anderen."""
    import inspect

    src = inspect.getsource(cc.layer_census_from_headers)
    assert "exclude_prefixes" in src and "exclude_segments" in src
    i_pre = src.index("exclude_prefixes and str(name).startswith")
    i_seg = src.index("exclude_segments and any(")
    assert i_pre < i_seg, "beide Klauseln stehen nacheinander, keine ersetzt die andere"
