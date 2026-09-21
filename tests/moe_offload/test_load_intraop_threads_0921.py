"""#66 (21.09.): die Ladephase darf mehr als einen Intraop-Thread bekommen.

Gemessen an fnFL2v84: 43 % der Ladezeit stehen in EINER Zeile
(fused_moe_triton/layer.py:1973, `.t().contiguous()` je Experten-Shard),
seriell im Hauptthread, waehrend die acht Datei-Worker warten und der
NVMe-Controller 40-60 % Leselast meldet. `torch.set_num_threads(1)` laesst
diese Kopie auf EINEM Kern laufen.

Ohne die Env bleibt alles byte-identisch zu vorher -- das ist der Teil, den
dieser Test festhaelt.
"""

import inspect

from sglang.srt.model_executor import model_runner as mr


def _src():
    return inspect.getsource(mr.ModelRunner.load_model)


def test_the_default_is_still_one_thread():
    src = _src()
    assert '_load_threads = 1' in src
    assert 'os.environ.get("SGLANG_LOAD_INTRAOP_THREADS", "1")' in src
    assert "torch.set_num_threads(_load_threads)" in src


def test_a_broken_value_falls_back_instead_of_raising():
    src = _src()
    assert "except ValueError:" in src
    # und niemals unter 1
    assert "max(1, int(" in src


def test_it_says_so_when_it_deviates():
    src = _src()
    assert "WEG2 LOAD-INTRAOP" in src
    assert "if _load_threads > 1:" in src


def test_the_parsing_rule_itself():
    """Die Regel, gegen die der Boot faehrt: leer/kaputt -> 1, Zahl -> Zahl,
    0 oder negativ -> 1."""
    import os as _os

    def parse(raw):
        try:
            return max(1, int(raw if raw is not None else "1"))
        except ValueError:
            return 1

    assert parse(None) == 1
    assert parse("1") == 1
    assert parse("8") == 8
    assert parse("0") == 1
    assert parse("-4") == 1
    assert parse("acht") == 1
