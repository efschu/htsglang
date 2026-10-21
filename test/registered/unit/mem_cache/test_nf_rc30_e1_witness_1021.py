"""E1-WITNESS (NF rc30 1d822bd150, boot 1021_063252): the e1 ANCHOR-ONLY READ line shared its counter with the
ANCHOR-ONLY BACKUP line (``_pdflip_anchor_only_n``), so after 16 backups no read was printed (rc30 D TP0: ONE read line,
n=13, pdflip-2-17 depth=23552 attached=0 -- a branch point, reached == branching). The read now counts alone, every miss
is printed and named like S109R's (held / written_here / branch_point_never_anchored / store_miss / held_unread)."""
import importlib.util
import logging
import os
import pathlib
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
LOGGER = "flliper.srt.mem_cache.unified_radix_cache"


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


E1 = _load("_e1_fixture_for_witness", _HERE / "test_anchor_only_read_e1_1014.py")


def test_a_miss_after_many_backups_is_still_printed_and_named(monkeypatch, caplog):
    """RED on 71caa1d932: with the shared counter past 16 the miss is silent."""
    caplog.set_level(logging.INFO, LOGGER)
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_anchor_only_n", 500, raising=False)   # 500 backups printed before
    be = E1._Backend(ok=False)
    t, node, pool = E1._tree(backend=be, monkeypatch=monkeypatch)
    req = E1._req(rid="pdflip-2-17", mamba_branching_seqlen=E1.DEPTH)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(req) == 0
    lines = [m for m in caplog.messages if "PDFLIP ANCHOR-ONLY-READ" in m]
    assert lines, caplog.messages
    assert "attached=0" in lines[-1] and "how=branch_point_never_anchored" in lines[-1] and "defect=0" in lines[-1]


def test_an_attach_keeps_its_line_and_is_no_defect(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_aor_read_n", 0, raising=False)
    be = E1._Backend(ok=True, row=2)
    t, node, pool = E1._tree(backend=be, monkeypatch=monkeypatch)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(E1._req()) == E1.DEPTH
    line = [m for m in caplog.messages if "PDFLIP ANCHOR-ONLY-READ" in m][-1]
    assert "attached=1" in line and "how=attach" in line and "defect=0" in line
