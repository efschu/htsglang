"""#142: JEDER Device-Lesezugriff im Abort-Poll braucht seinen eigenen Guard.

Die Wurzel von fnFL2w73/w74 war nicht, dass EIN Guard fehlte, sondern dass
der #1330-Guard fuer `_ctl_dev` geschrieben wurde und der #622-Rundenspiegel
danach IN SEINEN `with`-Block gesetzt wurde, ohne ihn mitzunehmen. Ein Test
auf genau `_round_dev` wuerde denselben Fehler beim naechsten Puffer wieder
durchlassen -- also prueft dieser Test die REGEL: fuer jeden Tensor, aus dem
der Poll auf dem Device liest, muss im selben Funktionskoerper ein
`data_ptr()`-Vergleich stehen.

Der Test laeuft ohne Karte (AST ueber die Quelle), weil die Alternative --
die Methode an einer echten BarlinkBar1-Instanz fahren -- eine GPU, einen
initialisierten BAR1-Transport und einen PAUSIERTEN TMS-Puffer braucht; das
ist ein Boot, kein Test. Was hier geprueft wird, ist die Eigenschaft, die
beim Schreiben des naechsten Lesers verletzt wird.
"""
import ast
import pathlib

SRC = (pathlib.Path(__file__).resolve().parents[2] / "python/sglang/srt"
       / "distributed/device_communicators/barlink_bar1.py")


def _poll_body():
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "poll_status_word":
            return node
    raise AssertionError("poll_status_word nicht gefunden")


def _guarded_names(fn):
    """Namen, zu denen im Koerper ein `<name>...data_ptr()`-Vergleich steht."""
    out = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare):
            continue
        for side in [node.left, *node.comparators]:
            for sub in ast.walk(side):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "data_ptr"):
                    for a in ast.walk(side):
                        if isinstance(a, ast.Attribute) and a.attr.startswith("_"):
                            out.add(a.attr)
    return out


def _device_sources(fn):
    """self._x, aus denen ein copy_() liest (das Argument, nicht das Ziel)."""
    out = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy_"
                and node.args):
            for a in ast.walk(node.args[0]):
                if (isinstance(a, ast.Attribute)
                        and isinstance(a.value, ast.Name)
                        and a.value.id == "self"
                        and a.attr.endswith("_dev")):
                    out.add(a.attr)
    return out


def test_jede_device_quelle_im_poll_ist_geguarded():
    fn = _poll_body()
    quellen = _device_sources(fn)
    assert quellen, "der Test findet die Lesezugriffe nicht mehr -- Poll umgebaut?"
    fehlend = quellen - _guarded_names(fn)
    assert not fehlend, (
        f"ungeschuetzte Device-Lesezugriffe im Abort-Poll: {sorted(fehlend)}. "
        "Ein Puffer, dessen TMS-Mapping die Phase genommen hat, hat "
        "data_ptr()==0; copy_ darauf faultet im Treiber, in einem "
        "POLL-THREAD -- der Rang stirbt sichtbar woanders (w73/w74: im "
        "Marlin-Repack des Drafts). Guard an den Lesezugriff, nicht an den "
        "Puffer, fuer den er damals geschrieben wurde.")


def test_beide_bekannten_puffer_sind_erfasst():
    """Regression: _ctl_dev (#1330) und _round_dev (#142, w73/w74)."""
    quellen = _device_sources(_poll_body())
    assert {"_ctl_dev", "_round_dev"} <= quellen, sorted(quellen)
