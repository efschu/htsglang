"""S109R: an anchor the S109c aux round released (its L3 copy proven) is read back from L3 at the next hit.

THE CLASS (NF rc29 a773b4ebc4, follow-up of the H105g death 02:31:13Z). ``_s109c_aux_device_round`` (81037f0c40)
releases, from a DEVICE-KV node whose only host-only aux state is its mamba anchor, the Full host rows AND the anchor
(tree law: aux host first) -- only once the write-behind's record holds the L3 copy of every KV page and of the anchor
(``secure_rows_to_l3`` + ``l3_complete``, ``_secure_child_anchors`` None). The node stays: Full device value, no Full
host, no mamba state anywhere (D log: ``S109c W3-AUX-DEVICE SPILL`` 6x on TP0, 02:25:49-02:29:33Z).

The next hit on that node: the match refuses it (``#904 ... MambaComponent:absent``) and falls back to the last state
above; e1 (``pdflip_anchor_only_read``) asked the store -- and returned 0 SILENTLY for exactly this shape
(AUX-HOST-LAW: no host state on a node without a Full host copy). Every token behind the last state was recomputed
although L3 held the anchor; e1 was asked only at intake and at the deferral retry anyway.

WHAT MUST HOLD.
(1) After a real S109c release, e1 reads the anchor from the store into a DEVICE mamba slot (S90b's device attach),
    witnessed ``S109R ANCHOR-DEVICE-READ ... attached=1`` (RED on a773b4ebc4: 0, no read, no line).
(2) Where peers hold mamba replicas, no rank-local device slot is taken (#639b), named ``how=peer_replicas``
    (mutant: guard removed -> it reads).
(3) A store miss is witnessed (``how=store_miss attached=0``) -- the monitor's count of recomputed-despite-L3.
(4) The ADMISSION match asks too (group D only), and a re-match follows an attach.
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import pathlib
import sys
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.managers import scheduler as sched_mod  # noqa: E402
from flliper.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import forma_l3_subtree as FLS  # noqa: E402
from flliper.srt.pdflip import host_only_spill as HOS  # noqa: E402

_UNIT = pathlib.Path(__file__).resolve().parent.parent
FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
LOGGER = "flliper.srt.mem_cache.unified_radix_cache"


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


S90B = _load("_s90b_fixture_for_s109r", _UNIT / "pdflip" / "test_s90b_anchor_l3_1020.py")
S109 = _load("_s109_fixture_for_s109r", _UNIT / "mem_cache" / "test_s109_backup_wall_aux_leaves_1020.py")


def _s109c_release(t, node):
    """Run the REAL ``_s109c_aux_device_round`` on ``node`` (TP0 of a Form A group, KV pages and anchor on L3)."""
    t.components[FULL] = object()
    t.evictable_host_leaves = types.SimpleNamespace(discard=lambda n: None)   # fixture nodes are unhashable
    t._w3_spill_keys = lambda pool, nodes: S109._Keys({id(n): (0, getattr(n, "id", 0)) for n in nodes})

    def _evict(n, comp, target=None, tracker=None):   # the tree law of _evict_aux_host_before_full_host
        k = int(n.component_data[FULL].host_value.numel())
        n.component_data[MAMBA].host_value = None
        n.component_data[FULL].host_value = None
        return 0, k

    t._evict_component_and_detach_lru = _evict
    with mock.patch.object(FLS, "_secure_child_anchors", lambda tree, order, budget: None), \
            mock.patch.object(R12, "record_state", lambda tree, n, why=None: None):
        out = UnifiedRadixCache._s109c_aux_device_round(t, S109._KvPool(), [node], 1, "host", 1)
    assert out["spilled"] == 1, out
    return out


def _world(monkeypatch, *, ok=True):
    be = S90B._Backend(ok=ok)
    t, n1, n2, hp, dev = S90B._ftree(monkeypatch, be, full_host=True)
    # S109c's candidate: device KV + Full host rows, the END anchor host-only (arena slot 5)
    n1.component_data[MAMBA].host_value = torch.tensor([5], dtype=torch.int64)
    n1.l3_present = False
    _s109c_release(t, n1)
    cdf, cdm = n1.component_data[FULL], n1.component_data[MAMBA]
    assert cdf.value is not None and cdf.host_value is None and n1.l3_present
    assert cdm.value is None and cdm.host_value is None, "S109c released the anchor (its L3 copy proven)"
    return t, n1, be, hp, dev


def _req(**kw):
    base = dict(rid="pdflip-2-50", key_match_depth=S90B.D_TOLD, state_anchor_depth=0,
                full_untruncated_fill_ids=S90B.TOKENS, extra_key=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_after_a_real_s109c_release_the_hit_reads_the_anchor_from_l3_into_a_device_slot(monkeypatch, caplog):
    """RED on a773b4ebc4: e1 returns 0 without reading (AUX-HOST-LAW) -- the match stays at the last state above and
    the tokens behind it are recomputed. GREEN: one blob read by the node's last page hash, copied into device slot 7."""
    caplog.set_level(logging.INFO, LOGGER)
    t, n1, be, hp, dev = _world(monkeypatch)
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: True)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == S90B.D_TOLD
    assert be.calls == [(PoolName.MAMBA, ["n19200-last"])]
    cd = n1.component_data[MAMBA]
    assert cd.value is not None and int(cd.value[0]) == 7
    assert cd.host_value is None, "AUX-HOST-LAW kept: no host state on a node without a Full host copy"
    assert hp.loads == [0, 1] and hp.freed == [5]
    assert any("S109R ANCHOR-DEVICE-READ" in m and "attached=1" in m and "how=device_attach" in m
               for m in caplog.messages), caplog.messages


def test_peer_replicas_never_take_a_rank_local_device_slot(monkeypatch, caplog):
    """#639b: a TP group whose peers hold mamba replicas -- a device slot (and the eviction behind it) on one rank
    would split the tombstone sets. Named, not taken."""
    caplog.set_level(logging.INFO, LOGGER)
    t, n1, be, hp, _ = _world(monkeypatch)
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: False)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == 0
    assert be.calls == [] and n1.component_data[MAMBA].value is None
    assert any("S109R ANCHOR-DEVICE-READ" in m and "how=peer_replicas" in m for m in caplog.messages)


def test_mutant_without_the_rank_alone_guard_reads_on_a_replicated_group(monkeypatch):
    """The guard above is what keeps the replicated group equal: removed, the same inputs take a slot."""
    t, n1, be, _, _ = _world(monkeypatch)
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: False)
    monkeypatch.setattr(UnifiedRadixCache, "_s109r_rank_alone", lambda self: True)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == S90B.D_TOLD
    assert be.calls and n1.component_data[MAMBA].value is not None


def test_a_store_miss_is_witnessed_for_the_monitor(monkeypatch, caplog):
    caplog.set_level(logging.INFO, LOGGER)
    t, n1, be, hp, _ = _world(monkeypatch, ok=False)
    monkeypatch.setattr(HOS, "form_a_host", lambda cache: True)
    with envs.FLLIPER_PDFLIP_ANCHOR_ONLY_READ.override(True):
        assert t.pdflip_anchor_only_read(_req()) == 0
    assert n1.component_data[MAMBA].value is None and hp.freed == [10_000]
    assert any("S109R ANCHOR-DEVICE-READ" in m and "how=store_miss" in m and "attached=0" in m
               for m in caplog.messages), caplog.messages


def test_the_s109c_line_names_the_released_nodes(monkeypatch, caplog):
    """The pair the monitor counts: ``S109c ... ids=<id>@<depth>`` then ``S109R ... depth=<depth>``."""
    caplog.set_level(logging.INFO, LOGGER)
    _world(monkeypatch)
    lines = [m for m in caplog.messages if "S109c W3-AUX-DEVICE SPILL" in m]
    assert lines and "ids=n19200@%d" % S90B.D_TOLD in lines[-1], lines


# ---- (4) the admission match asks too -------------------------------------------------------------------------------

def _sched(pp_size, read_depth):
    tc = types.SimpleNamespace(pdflip_anchor_only_read=lambda req: read_depth)
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=pp_size), tree_cache=tc)


def test_admission_read_rematches_after_an_attach_on_group_d_only():
    req = mock.MagicMock()
    assert sched_mod._s109r_admission_anchor_read(_sched(1, 192), req) is True
    req.init_next_round_input.assert_called_once()
    req2 = mock.MagicMock()
    assert sched_mod._s109r_admission_anchor_read(_sched(1, 0), req2) is False
    req2.init_next_round_input.assert_not_called()
    req3 = mock.MagicMock()
    assert sched_mod._s109r_admission_anchor_read(_sched(3, 192), req3) is False   # PP form: PP0's told reads
    req3.init_next_round_input.assert_not_called()


def test_the_admission_loop_asks_right_after_its_match():
    """Mutant guard: the call sits directly behind the admission match of ``_get_new_batch_prefill_raw`` (removing it
    leaves a waiting request that lost its anchor to S109c recomputing)."""
    src = inspect.getsource(sched_mod.Scheduler._get_new_batch_prefill_raw)
    i = src.index("req.init_next_round_input(self.tree_cache)\n")
    j = src.index("_s109r_admission_anchor_read(self, req)")
    assert 0 < j - i < 400, (i, j)
