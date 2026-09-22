"""#113: der exportierbare VMM-Handle -- Leser UND Schreiber.

Die Union-Arena (500c984795, METALL-BEWIESEN auf der 5090) teilt ein
Gewichtsbild zwischen den Phasen-Prozessen. Sie erreichte bisher nur ihre
eigene Arena (1,50 GiB), weil `tms_csrc/utils.h` jede Allokation OHNE
`requestedHandleTypes` anlegt -- dann kann `cuMemExportToShareableHandle`
auf ihr nicht gelingen (understand_prior-art.md Sec.5).

Die Tests binden BEIDE Seiten der Naht: den C++-Leser und den Env-Schreiber
im Launcher. Genau diese Gegenprobe hat heute zweimal gefehlt.
"""
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "sglang"
_UTILS_ROH = (_ROOT / "srt/weg2/tms_csrc/utils.h").read_text()
_LAUNCHER_ROH = (_ROOT / "srt/weg2/launcher.py").read_text()


def _ohne_kommentare(text: str, marker: str) -> str:
    """Nur der CODE. Die erste Fassung dieser Tests fand jeden Begriff im
    ERKLAERTEXT und blieb gruen, waehrend der Code etwas anderes tat -- am
    selben Tag zum zweiten Mal (siehe test_repack_adoption_0922)."""
    return "\n".join(
        z for z in text.split("\n") if not z.lstrip().startswith(marker)
    )


_UTILS = _ohne_kommentare(_UTILS_ROH, "//")
_LAUNCHER = _ohne_kommentare(_LAUNCHER_ROH, "#")

ENV = "SGLANG_WEG2_VMM_EXPORTABLE"


def test_der_leser_existiert():
    assert ENV in _UTILS
    assert "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR" in _UTILS


def test_der_schreiber_existiert():
    # Die Klasse, die heute zehnmal auftrat: ein Leser, den niemand bedient.
    assert f'xchg_env["{ENV}"]' in _LAUNCHER


def test_die_env_geht_an_BEIDE_gruppen():
    # xchg_env erreicht P und D. Stuende sie in env_p oder env_d, koennte der
    # Besitzer exportierbar anlegen und der Peer trotzdem nicht importieren.
    i = _LAUNCHER.index(f'xchg_env["{ENV}"]')
    assert "xchg_env" in _LAUNCHER[i - 200 : i]


def test_default_aus():
    # Ohne die Env muss die Funktion byte-identisch zu vorher sein.
    i = _UTILS.index(ENV)
    block = _UTILS[i : i + 400]
    assert "[0] == '1'" in block, "die Env wird nicht auf '1' geprueft"


def test_ein_refused_export_toetet_den_boot_nicht():
    # Ein Feature, das sich nicht anschalten laesst, darf kein Feature sein,
    # das nicht bootet.
    i = _UTILS.index(ENV)
    block = _UTILS[i : i + 1200]
    assert "CUDA_SUCCESS" in block and "falling back" in block
    # und der Rueckfall nimmt die ALTE prop, nicht die exportierbare
    assert "cuMemCreate(alloc_handle, size, &prop, 0)" in block
