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

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "flliper"
_UTILS_ROH = (_ROOT / "srt/pdflip/tms_csrc/utils.h").read_text()
_LAUNCHER_ROH = (_ROOT / "srt/pdflip/launcher.py").read_text()


def _strip_comments(text: str, marker: str) -> str:
    """Nur der CODE. Die erste Fassung dieser Tests fand jeden Begriff im
    ERKLAERTEXT und blieb gruen, waehrend der Code etwas anderes tat -- am
    selben Tag zum zweiten Mal (siehe test_repack_adoption_0922)."""
    return "\n".join(
        z for z in text.split("\n") if not z.lstrip().startswith(marker)
    )


_UTILS = _strip_comments(_UTILS_ROH, "//")
_LAUNCHER = _strip_comments(_LAUNCHER_ROH, "#")

ENV = "FLLIPER_PDFLIP_VMM_EXPORTABLE"


def test_reader_exists():
    assert ENV in _UTILS
    assert "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR" in _UTILS


def test_writer_exists():
    # Die Klasse, die heute zehnmal auftrat: ein Leser, den niemand bedient.
    assert f'xchg_env["{ENV}"]' in _LAUNCHER


def test_env_goes_to_both_groups():
    # xchg_env erreicht P und D. Stuende sie in env_p oder env_d, koennte der
    # Besitzer exportierbar anlegen und der Peer trotzdem nicht importieren.
    i = _LAUNCHER.index(f'xchg_env["{ENV}"]')
    assert "xchg_env" in _LAUNCHER[i - 200 : i]


def test_default_aus():
    # Ohne die Env muss die Funktion byte-identisch zu vorher sein.
    i = _UTILS.index(ENV)
    block = _UTILS[i : i + 400]
    assert "[0] == '1'" in block, "the Env is not checked against '1'"


def test_refused_export_does_not_kill_the_boot():
    # Ein Feature, das sich nicht anschalten laesst, darf kein Feature sein,
    # das nicht bootet.
    i = _UTILS.index(ENV)
    block = _UTILS[i : i + 1200]
    assert "CUDA_SUCCESS" in block and "falling back" in block
    # und der Rueckfall nimmt die ALTE prop, nicht die exportierbare
    assert "cuMemCreate(alloc_handle, size, &prop, 0)" in block
