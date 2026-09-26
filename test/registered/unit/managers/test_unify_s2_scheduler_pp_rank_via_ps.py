"""UNIFY S2 (Draft-Zensus self.ps): EINE Form fuer einen doppelt gebauten Fix.

Beide Linien haben denselben Befund unabhaengig gefixt, je an ihrer eigenen Zeile:

* 27B 0ab2687668 (xsn416): die DFLASH-PRODUCE-ON-P-off-Zeile in
  ``_maybe_init_draft_kv_producer`` las ``self.pp_rank`` -> AttributeError, PP2 tot vor READY.
* NF cc660d8786: der #161-Zensus ``pp{..}-draft-vs-target`` nach dem Draft-Load las
  ``self.pp_rank`` -> AttributeError, vom umgebenden except verschluckt, die Zeile loggte auf P nie.

Die Absicht beider Seiten ist dieselbe Invariante: der Scheduler traegt seinen PP-Rang als
``self.ps.pp_rank``, ein ``self.pp_rank`` gibt es auf ihm nicht. Statt zweier Zeilen-Pins (27B pinnt
nur die Producer-Funktion, NF hatte keinen Pin) pinnt dieser Test die Invariante fuer die ganze
Scheduler-Klasse samt ihrer sglang-Mixins. Damit ist auch die 27B-Zeile abgedeckt, sobald
``--dflash-produce-on-p`` in Schritt 6 in den Integrationszweig kommt.
"""

import ast
import inspect
import textwrap


def _self_pp_rank_reads(cls):
    src = textwrap.dedent(inspect.getsource(cls))
    tree = ast.parse(src)
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == "pp_rank"
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
    ]


def test_scheduler_and_its_mixins_never_read_self_pp_rank():
    from sglang.srt.managers.scheduler import Scheduler

    checked = []
    bad = {}
    for cls in Scheduler.__mro__:
        if not cls.__module__.startswith("sglang."):
            continue
        checked.append(f"{cls.__module__}.{cls.__qualname__}")
        hits = _self_pp_rank_reads(cls)
        if hits:
            bad[f"{cls.__module__}.{cls.__qualname__}"] = hits
    assert "sglang.srt.managers.scheduler.Scheduler" in checked
    assert not bad, (
        f"self.pp_rank read in {bad} (relative lines); the Scheduler carries it as "
        "self.ps.pp_rank (27B xsn416 / NF #161-Zensus)"
    )


def test_scheduler_has_no_pp_rank_attribute_to_fall_back_on():
    """Die Invariante stimmt nur, solange niemand ``self.pp_rank`` still nachruestet: sonst
    waere der Pin oben eine Namensfrage und keine Absturzfrage mehr."""
    from sglang.srt.managers.scheduler import Scheduler

    for cls in Scheduler.__mro__:
        if not cls.__module__.startswith("sglang."):
            continue
        assert "pp_rank" not in vars(cls), f"{cls.__qualname__} defines pp_rank"
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        writes = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            if isinstance(t, ast.Attribute)
            and t.attr == "pp_rank"
            and isinstance(t.value, ast.Name)
            and t.value.id == "self"
        ]
        assert not writes, f"{cls.__qualname__} writes self.pp_rank at {writes}"
