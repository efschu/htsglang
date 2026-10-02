"""L15 desk repro (N5n 14:44:28 L15-CHECK-DIAG foreign=64 on ranks 1/2): after
retain's compact moves and the tree rewrite, every compact row the wake check
samples (owned_l2_rows of the manifest) must hold the SAME token whose L2
identity the manifest pairs with it. Device rows carry a per-row marker, the
L2 identity is distinct per token (300 + token index); the check pairing is
followed back to the token's ORIGINAL device row.

A foreign pairing here (manifest row X <-> L2 of token Y) would reproduce the
metal's foreign=64 with equal magnitudes (live and L2 are both real KV).
Geometry and fakes from test_weg2_l15_retain_0930 (prefix (0,1,2), rank 1,
slot 9 compacts onto free slot 5)."""
import importlib.util
import os

import torch

from sglang.srt.weg2 import l15_restore, l15_retain
from sglang.srt.weg2.l15_manifest import read as manifest_read

_RT = os.path.join(os.path.dirname(__file__), "test_weg2_l15_retain_0930.py")


def _rt():
    spec = importlib.util.spec_from_file_location("test_weg2_l15_retain_0930", _RT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _local_row(prefix, rank, slot):
    return l15_restore._compact_row(list(prefix), rank, int(slot))


def test_compact_moves_keep_the_row_l2_pairing(tmp_path):
    rt = _rt()
    sc = rt.make_scenario(tmp_path, [])
    kw = dict(sc["kwargs"])
    l2 = {rid: tuple(300 + i for i in range(len(s))) for rid, s in rt.SLOTS_OF.items()}
    kw["l2_of"] = lambda rid: (l2[rid], tuple(1 for _ in l2[rid]))
    kv0 = sc["kv_buf"].clone()
    res = l15_retain.retain_at_sleep(**kw)
    assert res is not None
    m = manifest_read(sc["manifest_path"], pid_alive=lambda _pid: True)
    rows = l15_restore.owned_l2_rows(m, rt.RANK, list(rt.PREFIX))
    assert rows, "the check would sample nothing"
    seen = 0
    for row, l2_slot, _gen, _lane, rids in rows:
        rid = rids[0]
        i = int(l2_slot) - 300
        orig_slot = rt.SLOTS_OF[rid][i]
        orig_row = _local_row(rt.PREFIX, rt.RANK, orig_slot)
        assert torch.equal(sc["kv_buf"][row], kv0[orig_row]), (
            f"row {row} carries a foreign token for {rid}[{i}] "
            f"(orig slot {orig_slot} / row {orig_row})")
        seen += 1
    assert seen == len(rows)
