# SPDX-License-Identifier: Apache-2.0
"""SL (NF rc12u dkrnfh91dprtkbar1dauer09271905, pdflip-4-7 19:11:06): a follower that already holds the told
span was named "FOLLOWER REGISTRATION DECLINED ... cannot load what PP0 admitted" because
``_local_prefix`` took ``bool()`` of the torch tensor in ``prefix_indices`` (raises) and read 0.

PP1: "#1442 HANDOFF-KEYS REG matched=39232 new=1 span=0", "#915 PREFETCH REFUSED reason=too_short",
told 39232 -> the rank HOLDS the told span: satisfied locally, admission at told.
"""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers import pdflip_store_told as st  # noqa: E402


def _req(n_dev, host=0, rid="pdflip-4-7"):
    return types.SimpleNamespace(rid=rid, prefix_indices=torch.arange(n_dev), host_hit_length=host)


def _sched(verdict="declined:too_short"):
    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=1),
        tree_cache=types.SimpleNamespace(is_eagle=True),
        _pdflip_store_told={}, _pdflip_store_held={},
    )
    s.calls = []

    def _prefetch_kvcache(req, limit_tokens=None):
        s.calls.append(limit_tokens)
        return verdict

    s._prefetch_kvcache = _prefetch_kvcache
    return s


def test_local_prefix_reads_a_tensor():
    assert st._local_prefix(_req(39232)) == 39232
    assert st._local_prefix(_req(39000, host=232)) == 39232
    assert st._local_prefix(types.SimpleNamespace(rid="x", prefix_indices=None, host_hit_length=0)) == 0
    assert st._local_prefix(types.SimpleNamespace(rid="x")) == 0
    assert st._local_prefix(_req(1)) == 1


def test_rc12u_pdflip_4_7_follower_holding_the_told_span_is_satisfied(monkeypatch):
    monkeypatch.setattr(st, "_adopt_keys", lambda scheduler, req: None)
    s, r = _sched(), _req(39232)
    assert st._follower_register(s, r, 39232) == "satisfied:local_prefix"
    assert s._pdflip_store_told_satisfied == {"pdflip-4-7": 39232}
    assert s.calls == [39233]  # the bigram +1 limit, unchanged


def test_a_follower_short_of_told_still_declines(monkeypatch):
    monkeypatch.setattr(st, "_adopt_keys", lambda scheduler, req: None)
    s, r = _sched(), _req(39168)
    assert st._follower_register(s, r, 39232) == "declined:too_short"
    assert not getattr(s, "_pdflip_store_told_satisfied", {})


def test_admission_admits_a_satisfied_rid_at_told(monkeypatch):
    monkeypatch.setattr(st, "_adopt_keys", lambda scheduler, req: None)
    s, r = _sched(), _req(39232)
    st._follower_register(s, r, 39232)
    s._pdflip_store_told["pdflip-4-7"] = 39232
    s.tree_cache.check_prefetch_progress = lambda rid: (_ for _ in ()).throw(AssertionError("no wait"))
    assert st.admission(s, r, lambda *a: None) == 0
    assert "pdflip-4-7" not in s._pdflip_store_told
