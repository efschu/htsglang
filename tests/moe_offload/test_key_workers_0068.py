"""#68: der Pool teilt die Keys EINER Datei.

Der Docstring von `Qwen4ExpForConditionalGeneration.weight_post_load` nennt
den Grund, warum der Worker-Transpose (#66) am Metall LANGSAMER war (PP2
36,7 -> 47,5 s, fnFL2v87): "post_load runs serially per file". Auf diesem
Checkpoint sind das ~16 000 Tensoren je Shard in EINEM Thread.

Der Test prueft das ERGEBNIS an echten Bytes: seriell und parallel muessen
bit-identisch dasselbe liefern.
"""
import os
import struct
import json
import pathlib
import tempfile

import torch

from flliper.srt.model_loader import weight_utils as wu


def _write_safetensors(file_path, tensors):
    header, off = {}, 0
    roh = b""
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": {torch.float32: "F32", torch.int8: "I8"}[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [off, off + len(b)],
        }
        roh += b
        off += len(b)
    hj = json.dumps(header).encode()
    with open(file_path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(roh)


def _fixture(n=37):
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / "m.safetensors"
    tensors = {
        f"t{i}": (torch.arange(4 * (i + 1), dtype=torch.float32) + i).reshape(-1)
        for i in range(n)
    }
    _write_safetensors(p, tensors)
    return str(p), tensors


def test_parallel_is_bit_identical(monkeypatch):
    p, expected = _fixture()
    monkeypatch.delenv("FLLIPER_LOAD_KEY_WORKERS", raising=False)
    serial = wu.pread_safetensors_file(p)
    monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", "4")
    parallel = wu.pread_safetensors_file(p)
    assert set(serial) == set(parallel) == set(expected)
    for k in expected:
        assert torch.equal(serial[k], parallel[k]), k
        assert torch.equal(parallel[k], expected[k]), k


def test_should_load_gilt_auch_parallel(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", "4")
    nur = wu.pread_safetensors_file(p, should_load=lambda n: n in ("t1", "t5"))
    assert set(nur) == {"t1", "t5"}


def test_meta_gilt_auch_parallel(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", "4")
    d = wu.pread_safetensors_file(p, should_load=lambda n: "meta" if n == "t2" else True)
    assert d["t2"].device.type == "meta"
    assert d["t3"].device.type == "cpu"


def test_default_is_serial_form(monkeypatch):
    # Ohne die Env darf sich NICHTS aendern -- byte-identisch zu vorher.
    monkeypatch.delenv("FLLIPER_LOAD_KEY_WORKERS", raising=False)
    import inspect

    src = inspect.getsource(wu.pread_safetensors_file)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    i = code.index('FLLIPER_LOAD_KEY_WORKERS')
    assert '"1"' in code[i : i + 120], "der Default ist nicht 1"


def test_exception_is_not_swallowed(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", "4")

    def _bad(n):
        if n == "t7":
            raise RuntimeError("#68 probe")
        return True

    try:
        wu.pread_safetensors_file(p, should_load=_bad)
    except RuntimeError as e:
        assert "#68 probe" in str(e)
    else:
        raise AssertionError("die Ausnahme wurde verschluckt -- ein halb "
                             "gelesener Gewichtssatz laedt still falsch")


def test_68b_post_load_runs_in_key_thread(monkeypatch):
    """Die Key-Parallelitaet allein brachte NICHTS (w56 106,52 s gegen w54
    110,16 bei 30-50 % NVMe-Last): gelesen wurde parallel, aber `post_load`
    lief danach seriell beim Aufrufer ueber ~16 000 Keys. Hier laeuft es
    im lesenden Thread."""
    import threading

    p, _ = _fixture(n=40)
    monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", "4")
    threads = set()

    def _pl(name, t):
        threads.add(threading.get_ident())
        return t * 2

    d = wu.pread_safetensors_file(p, post_load=_pl)
    assert len(threads) > 1, (
        f"post_load lief in {len(threads)} Thread(s) -- die CPU-Arbeit ist "
        f"weiter seriell, genau der Befund aus w56"
    )
    # und es wurde GENAU EINMAL je Tensor angewandt (nicht zweimal)
    expected = wu.pread_safetensors_file(p)
    for k in expected:
        assert torch.equal(d[k], expected[k] * 2), k


def test_68b_no_double_post_load_in_pread_path():
    import inspect

    src = inspect.getsource(wu.buffered_multi_thread_safetensors_weights_iterator)
    i_pread = src.index("post_load=post_load, direct_io=direct_io")
    i_serial = src.index("{k: post_load(k, v) for k, v in result.items()}")
    assert i_pread < i_serial
    assert "return _erg" in src[i_pread : i_pread + 200], (
        "der pread-Pfad faellt in die serielle Schleife durch -- post_load "
        "liefe zweimal und das Modell laedt still falsch"
    )


def test_68c_serial_branch_applies_post_load(monkeypatch):
    """w59-Wurzel: `_load_file` ueberspringt seit #68b seine eigene
    post_load-Schleife fuer JEDEN pread-Pfad. Der serielle Zweig wandte
    post_load aber nicht an -- ohne FLLIPER_LOAD_KEY_WORKERS>1 kam jeder
    Experten-Shard untransponiert beim Lader an (PP2 starb nach 7,3 s an
    'size of tensor a (2560) must match the size of tensor b (80)')."""
    p, expected = _fixture(n=9)
    monkeypatch.delenv("FLLIPER_LOAD_KEY_WORKERS", raising=False)
    d = wu.pread_safetensors_file(p, post_load=lambda n, t: t * 2)
    assert set(d) == set(expected)
    for k in expected:
        assert torch.equal(d[k], expected[k] * 2), (
            f"{k} kam UNVERAENDERT zurueck -- post_load lief im seriellen "
            f"Zweig nicht"
        )


def test_68c_once_per_tensor_in_both_branches(monkeypatch):
    """Die Zaehlprobe ueber den ECHTEN Verbraucherpfad (`_load_file` im
    buffered iterator): jeder Tensor sieht post_load genau einmal -- egal
    ob der serielle oder der parallele Zweig liest. Zweimal waere ein
    stilles Falschladen, keinmal der w59-Tod."""
    p, expected = _fixture(n=11)
    for kw in (None, "4"):
        if kw is None:
            monkeypatch.delenv("FLLIPER_LOAD_KEY_WORKERS", raising=False)
        else:
            monkeypatch.setenv("FLLIPER_LOAD_KEY_WORKERS", kw)
        counter = {}

        def _pl(name, t):
            counter[name] = counter.get(name, 0) + 1
            return t * 2

        seen = dict(
            wu.buffered_multi_thread_safetensors_weights_iterator(
                [p], max_workers=2, pread=True, post_load=_pl
            )
        )
        assert set(seen) == set(expected), kw
        for k in expected:
            assert counter[k] == 1, f"kw={kw}: {k} sah post_load {counter[k]}x"
            assert torch.equal(seen[k], expected[k] * 2), f"kw={kw}: {k}"
