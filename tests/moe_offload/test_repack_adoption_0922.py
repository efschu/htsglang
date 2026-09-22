"""#111/#112: der Marlin-Repack beim Laden.

#112 ist die WURZEL von fnFL2w53 UND w54. Beide starben mit
`Fatal Python error: Segmentation fault`, der Stack zeigte
`gptq_marlin_repack.py:37 <- gptq_kernels.py:81`, und in w53 traf es
zusaetzlich einen Nebenthread -- was die Diagnose zuerst auf ein
Instrument lenkte. Eine Ursache, zwei Opfer: `initialize_dummy_weights`
fuellt auch `*_g_idx_sort_indices` mit Zufall, und der Repack benutzt sie
als INDIZES.
"""
import pathlib

# Die Quelltexte werden GELESEN, nicht importiert: `gptq_kernels` direkt zu
# importieren loest einen vorbestehenden Zirkelimport aus (er wird sonst nur
# ueber die Quantisierungs-Registry erreicht). Ein Test, der das Modul als
# erster anfasst, pruefte den Zirkel statt den Fix.
_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "sglang"


def _quelle(rel: str) -> str:
    return (_ROOT / rel).read_text()


_KERNELS = _quelle("srt/hardware_backend/gpu/quantization/gptq_kernels.py")
_SCHEME = _quelle(
    "srt/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_wNa16_moe.py"
)


def test_112_der_repack_laeuft_nicht_auf_platzhaltern():
    src = _SCHEME
    i_gate = src.index("weights_are_placeholder")
    i_repack = src.index("gptq_marlin_moe_repack(")
    assert i_gate < i_repack, (
        "der Repack steht vor seinem Riegel -- dann indiziert er wieder mit "
        "Zufallswerten und nimmt den CUDA-Kontext mit"
    )


def test_112_die_struktur_entsteht_trotzdem():
    # Der Presplit und die Pool-Geometrie haengen an den marlin-geformten
    # Puffern. Wird unter Adoption gar nichts gesetzt, entsteht eine ANDERE
    # Geometrie als auf dem Plattenweg -- und der Store passt nicht mehr.
    src = _SCHEME
    zweig = src[src.index("weights_are_placeholder"):src.index("else:", src.index("weights_are_placeholder"))]
    assert 'replace_tensor("w13_weight_packed"' in zweig
    assert 'replace_tensor("w2_weight_packed"' in zweig


def test_111_kein_zwischenpuffer_und_kein_lookup_je_experte():
    src = _KERNELS[_KERNELS.index("def gptq_marlin_moe_repack"):_KERNELS.index("class GPTQLinearKernel")]
    # OHNE KOMMENTARZEILEN. Die erste Fassung dieses Tests fand den
    # Modul-Lookup im ERKLAERTEXT ueber der Schleife und blieb gruen,
    # waehrend der Mutant ihn zurueck IN die Schleife setzte.
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    # Das Modul wird EINMAL geholt, vor der Schleife.
    i_mod = code.index("_jit_gptq_marlin_repack_module()")
    i_loop = code.index("for e in range(num_experts)")
    assert i_mod < i_loop, "der Modul-Lookup steht noch in der Schleife"
    assert code.count("_jit_gptq_marlin_repack_module()") == 1
    # Der Kernel schreibt DIREKT in output[e] -- keine Zuweisung aus einem
    # Rueckgabewert, also keine Allokation und keine Kopie je Experte.
    assert "output[e] = gptq_marlin_repack(" not in code
    assert "output[e], size_k, size_n, num_bits" in code


def test_111_der_wrapper_bleibt_fuer_andere_aufrufer():
    # Ein zweiter Aufrufer repackt einen EINZELNEN Tensor -- der Wrapper
    # darf nicht verschwinden, nur weil die MoE-Schleife ihn nicht mehr ruft.
    assert "x.data = gptq_marlin_repack(" in _KERNELS


def test_111_der_modul_getter_hat_denselben_fallback():
    # Sonst ist ein fehlgeschlagener Import ein NameError IN der Schleife
    # statt der Plattform-Meldung (Fehler im Fehlerpfad, #99).
    assert "_jit_gptq_marlin_repack_module = _unsupported_kernel" in _KERNELS


def test_112_die_leeren_puffer_liegen_auf_der_KARTE():
    """fnFL2w57: unter `dummy` liegt w13_weight_packed auf CPU, und die
    leeren Puffer erbten das -- `marlin_make_workspace` bekam dann
    `Expected a cuda device, but got: cpu`."""
    zweig = _SCHEME[_SCHEME.index("weights_are_placeholder"):_SCHEME.index("else:", _SCHEME.index("weights_are_placeholder"))]
    code = "\n".join(z for z in zweig.split("\n") if not z.lstrip().startswith("#"))
    assert "device=_dev" in code
    assert 'torch.device("cuda"' in code
    assert "device=layer.w13_weight_packed.device" not in code


def test_112_3_die_scales_gehen_auch_auf_die_karte():
    """create_weights baut JEDEN expert-major Tensor auf dem Host, wenn die
    Residenz-Fraction < 1.0 ist -- Scales und Zero-Points ebenso wie die
    packed weights. Der Adoptionszweig laeuft am device_loading_context des
    Loaders vorbei, also muessen sie hier selbst hinueber."""
    code = "\n".join(z for z in _SCHEME.split("\n") if not z.lstrip().startswith("#"))
    i_scales = code.index("marlin_w13_scales = marlin_moe_permute_scales")
    davor = code[max(0, i_scales - 900):i_scales]
    assert "w13_weight_scale" in davor and "_dev2" in davor
    assert "w13_weight_zero_point" in davor, "die Zero-Points fehlen"
    # und NUR unter Platzhaltern -- sonst eine Extrakopie ueber 48 Layer
    assert "if _platzhalter:" in davor
