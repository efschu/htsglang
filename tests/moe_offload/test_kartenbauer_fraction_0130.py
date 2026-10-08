"""#130: der Kartenbauer las die Residenz-Fraction aus der falschen Quelle.

`--pp-cut-expert-device-fraction` ist ein LAUNCHER-Flag und steht nie in
`--extra-p`. Der Ausdruck war immer leer, die Karte fiel auf ihre
Default-Residenz 188 zurueck: slots = 512-188 = 324, in JEDEM Boot
(gemessen fnFL2w60 bei FR 0.367 und fnFL2w62 bei FR 0.367,0.75,0.95 --
beide slots=324, shared_resident=92), waehrend der PP-Cut dieselben
Fractions korrekt sah.
"""
import inspect
import types

from flliper.srt.pdflip import launcher as L


def test_launcher_flag_is_read():
    assert L._split_fraction_text("0.367,0.75,0.95") == [0.367, 0.75, 0.95]
    assert L._split_fraction_text(" 0.5 ; 0.6 ") == [0.5, 0.6]


def test_empty_value_gives_empty_list():
    for leer in ("", None, "   ", "abc", "0.5,xx"):
        assert L._split_fraction_text(leer) == [], repr(leer)


def test_map_builder_does_not_ask_namespace_extra_p():
    src = inspect.getsource(L)
    i = src.index("_fr_p = ")
    window = src[i:i + 400]
    code = "\n".join(z for z in window.split("\n")
                     if not z.lstrip().startswith("#"))
    assert "pp_cut_expert_device_fraction" in code
    assert '_argv_vector(getattr(ns, "extra_p", ""),\n                             "--pp-cut-expert-device-fraction")' not in code, (
        "the card builder searches the launcher flag in extra_p again -- "
        "dort steht es nie, das war die w60/w62-Wurzel"
    )


def test_same_source_as_pp_cut():
    """Der PP-Cut liest ns.pp_cut_expert_device_fraction. Der Kartenbauer
    muss dieselbe Quelle nehmen, sonst laufen die beiden auseinander --
    die Klasse, die heute schon vier Boots gekostet hat."""
    src = inspect.getsource(L)
    assert src.count('getattr(ns, "pp_cut_expert_device_fraction"') >= 2
