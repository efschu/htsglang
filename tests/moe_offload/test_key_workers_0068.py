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

from sglang.srt.model_loader import weight_utils as wu


def _schreibe_safetensors(pfad, tensoren):
    header, off = {}, 0
    roh = b""
    for name, t in tensoren.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": {torch.float32: "F32", torch.int8: "I8"}[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [off, off + len(b)],
        }
        roh += b
        off += len(b)
    hj = json.dumps(header).encode()
    with open(pfad, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(roh)


def _fixture(n=37):
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / "m.safetensors"
    tensoren = {
        f"t{i}": (torch.arange(4 * (i + 1), dtype=torch.float32) + i).reshape(-1)
        for i in range(n)
    }
    _schreibe_safetensors(p, tensoren)
    return str(p), tensoren


def test_parallel_liefert_bitidentisch_dasselbe(monkeypatch):
    p, erwartet = _fixture()
    monkeypatch.delenv("SGLANG_LOAD_KEY_WORKERS", raising=False)
    seriell = wu.pread_safetensors_file(p)
    monkeypatch.setenv("SGLANG_LOAD_KEY_WORKERS", "4")
    parallel = wu.pread_safetensors_file(p)
    assert set(seriell) == set(parallel) == set(erwartet)
    for k in erwartet:
        assert torch.equal(seriell[k], parallel[k]), k
        assert torch.equal(parallel[k], erwartet[k]), k


def test_should_load_gilt_auch_parallel(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("SGLANG_LOAD_KEY_WORKERS", "4")
    nur = wu.pread_safetensors_file(p, should_load=lambda n: n in ("t1", "t5"))
    assert set(nur) == {"t1", "t5"}


def test_meta_gilt_auch_parallel(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("SGLANG_LOAD_KEY_WORKERS", "4")
    d = wu.pread_safetensors_file(p, should_load=lambda n: "meta" if n == "t2" else True)
    assert d["t2"].device.type == "meta"
    assert d["t3"].device.type == "cpu"


def test_default_ist_die_serielle_form(monkeypatch):
    # Ohne die Env darf sich NICHTS aendern -- byte-identisch zu vorher.
    monkeypatch.delenv("SGLANG_LOAD_KEY_WORKERS", raising=False)
    import inspect

    src = inspect.getsource(wu.pread_safetensors_file)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    i = code.index('SGLANG_LOAD_KEY_WORKERS')
    assert '"1"' in code[i : i + 120], "der Default ist nicht 1"


def test_eine_ausnahme_wird_nicht_verschluckt(monkeypatch):
    p, _ = _fixture()
    monkeypatch.setenv("SGLANG_LOAD_KEY_WORKERS", "4")

    def _boese(n):
        if n == "t7":
            raise RuntimeError("#68 probe")
        return True

    try:
        wu.pread_safetensors_file(p, should_load=_boese)
    except RuntimeError as e:
        assert "#68 probe" in str(e)
    else:
        raise AssertionError("die Ausnahme wurde verschluckt -- ein halb "
                             "gelesener Gewichtssatz laedt still falsch")
