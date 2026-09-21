"""#66 (21.09.): W11b muss sagen koennen, WO die unerklaerten Bytes sind.

fnFL2v86 verweigerte mit nvml_delta=5334,0 gegen resident 2202,6 +
head_released 1212,5 + tag_pool_inactive 1385,2 -- 533,7 MiB unerklaert bei
256 MiB Toleranz. Die Zeile konnte nicht sagen, ob der Rest INNERHALB von
torch liegt (ein Cache, den man benennen kann) oder AUSSERHALB (CUDA-Kontext,
cuBLAS-Workspaces, JIT-Kernel). Zwei Terme beantworten das -- als echte
Posten, nicht als weitere Toleranz.
"""

import re

from sglang.srt.weg2 import launcher as L


def _line(**kw):
    base = dict(
        resident_mib=2202.6, head_released_mib=1212.5, nvml_delta_mib=5334.0,
        tag_pool_inactive_mib=1385.2, outside_torch_mib=400.0,
        default_pool_inactive_mib=133.7,
    )
    base.update(kw)
    return (
        "[2026-09-21 13:41:36 PP2] WEG2 DRAFT-KV-PRODUCER armed stage=3/3 "
        "drafter=abc layout=v1 heads=2 head_dim=128 page_bytes=32768 "
        "embed=resident mtp_mib=1.0 embed_mib=2.0 "
        + " ".join(f"{k}={v}" for k, v in base.items())
        + " head_deferred=False embed_dtype=torch.int32 build_s=4.2"
    )


def test_the_two_new_terms_are_parsed():
    for name, rx in (
        ("outside_torch_mib", L._OUTSIDE_TORCH_RE),
        ("default_pool_inactive_mib", L._DEFAULT_POOL_RE),
    ):
        m = rx.search(_line())
        assert m is not None, name
    assert float(L._OUTSIDE_TORCH_RE.search(_line()).group(1)) == 400.0
    assert float(L._DEFAULT_POOL_RE.search(_line()).group(1)) == 133.7


def test_one_cache_term_never_both():
    """fnFL2v89, der Fehler beim ersten Versuch: torch.cuda.memory_reserved()
    zaehlt die privaten MemPools MIT. Gemessen torch_cached=2597,7 gegen
    pooled=1385,2 + released=1212,5 = 2597,7 auf die Nachkommastelle. Alle
    drei zu addieren trieb W11b auf -2064,0, das Spiegelbild der +533,7."""
    import inspect

    src = inspect.getsource(L)
    assert "delta - (r + torch_cached + outside)" in src
    assert "delta - (r + released + pooled + outside)" in src
    assert 'out["cache_term"] = "torch_total"' in src
    # nie alle drei zusammen
    assert "r + released + pooled + outside + " not in src
    # und die Toleranz ist NICHT angefasst worden
    assert "P_DRAFT_BUILD_ACCOUNTING_TOL_MIB" in src


def test_the_sum_identity_v89_measured():
    """Die Identitaet, die den Doppelzaehl-Fehler bewiesen hat."""
    pooled, released, torch_cached = 1385.2, 1212.5, 2597.7
    assert abs((pooled + released) - torch_cached) < 0.05


def test_an_unmeasured_term_is_zero_never_guessed():
    import inspect

    src = inspect.getsource(L)
    assert 'outside = 0.0 if outside is None or float(outside) < 0 else float(outside)' in src
    # der Gesamt-Cache faellt bei -1 auf die alte Rechnung zurueck, statt
    # als 0 eine Erklaerung vorzutaeuschen
    assert "if torch_cached >= 0:" in src


def test_the_arithmetic_of_v86_v89():
    """Was der Gesamt-Cache erklaert und was offen bleibt."""
    delta, r, torch_cached = 5334.0, 2202.6, 2597.7
    # mit dem Gesamt-Cache allein bleiben genau die 533,7 offen, die
    # outside_torch erklaeren soll -- und die in v89 NICHT messbar war (-1)
    assert abs(delta - (r + torch_cached) - 533.7) < 0.05
    # die fehlerhafte Addition aller drei kippte es auf -2064,0
    assert abs(delta - (r + 1212.5 + 1385.2 + 2597.7) + 2064.0) < 0.05


def test_the_emitter_prints_both():
    import inspect

    from sglang.srt.managers import scheduler as S

    src = inspect.getsource(S.Scheduler)
    assert "outside_torch_mib=%.1f default_pool_inactive_mib=%.1f" in src
    assert 'getattr(self.draft_kv_producer, "outside_torch_mib", -1.0)' in src
    assert 'getattr(self.draft_kv_producer, "default_pool_inactive_mib", -1.0)' in src


def test_the_producer_measures_before_and_after():
    import inspect

    from sglang.srt.speculative import draft_kv_producer as P

    src = inspect.getsource(P)
    assert "def _outside_torch_mib()" in src
    assert "def _default_pool_inactive_mib()" in src
    # der Import, der die Messung in v89 auf -1 fallen liess, ist weg
    assert "from sglang.srt.utils import get_device_id" not in src
    assert "self._outside_before_mib = _outside_torch_mib()" in src
    assert "self.outside_torch_mib = (" in src
    # ein Instrument faellt nie einen Boot
    assert src.count("# noqa: BLE001") >= 2
