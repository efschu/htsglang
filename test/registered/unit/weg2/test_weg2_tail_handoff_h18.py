"""fnFL2 H18 (E1 of H17): the tail hand-off P -> D, hermetic (no CUDA).

What these cases pin (all derived properties or bookkeeping a later diff can
silently break):
* the cut c = floor_r(N-1) with r the QSA compress ratio: c % r == 0 (the QSA
  device assert, fnFL2x14), 1 <= N-c <= r, the anchor page floor_64(c) is
  what the tree already holds, and no tail exists when c is page-aligned;
* the key is token-exact over [0, c) -- the needle safety of the hand-off;
* the group verdict is a MIN: one rank that cannot serve keeps the page resume;
* P's capture selects the GDN working slot and the partial page's rows (KV +
  QSA compressed groups) exactly, and D refuses a part whose digest, layer
  coverage or row shape does not match (D's adoption: test_weg2_tail_adopt_h21).
"""

import os
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize(
    "n, cut, prefix",
    [
        (97841, 97840, 97792),  # fnFL2x132 rid weg2-0-4: 49 -> 1 token extend
        (97856, 97852, 97792),  # N % 64 == 0: the cut is NOT N-64
        (97852, 97848, 97792),  # N % 4 == 0: the cut is N-4, never N
        (97795, 97792, 97792),  # cut page-aligned: no tail (node state suffices)
        (65, 64, 64),
    ],
)
def test_cut_on_the_compress_ratio(n, cut, prefix):
    assert th.tail_cut(n, 4) == cut
    assert cut % 4 == 0 and 1 <= n - cut <= 4
    assert th.page_floor(cut, 64) == prefix
    spec = th.spec_for("weg2-x", list(range(n)), None, 64, 4)
    if cut == prefix:
        assert spec is None
    else:
        assert (spec.page_prefix, spec.cut, spec.rows, spec.extend) == (prefix, cut, cut - prefix, n - cut)
        assert th.extend_range(spec) == (cut, n)


def test_grain_follows_the_switch_and_the_pool():
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True):
        assert th.anchor_grain(64, 4) == 4
        assert th.anchor_grain(64, None) == 1  # not QSA: unchanged
        assert th.anchor_grain(1, 4) == 1
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(False):
        assert th.anchor_grain(64, 4) == 64  # the x14 form, byte for byte
    # a page grain yields no spec at all (switch-off path writes nothing)
    assert th.spec_for("weg2-x", list(range(97841)), None, 64, 64) is None


def test_key_is_token_exact_below_the_cut_only():
    ids = list(range(1000, 1241))  # N=241, c=240
    base = th.tail_key(ids, 240, None)
    inside = list(ids)
    inside[200] += 1  # a token of the handed-over partial page
    assert th.tail_key(inside, 240, None) != base
    early = list(ids)
    early[3] += 1  # a token of the page prefix
    assert th.tail_key(early, 240, None) != base
    beyond = list(ids)
    beyond[240] += 1  # the token D recomputes anyway
    assert th.tail_key(beyond, 240, None) == base
    assert th.tail_key(ids, 240, "lora-a") != base


def test_group_verdict_is_a_min():
    assert th.agree_cut(97792, 97840, True, lambda v: v) == 97840  # single rank
    assert th.agree_cut(97792, 97840, True, lambda v: min(v, 1, 1)) == 97840
    # a Form-A worker answers 1; TP0 misses a part -> everyone stays at the page
    assert th.agree_cut(97792, 97840, True, lambda v: min(v, 0)) == 97792
    assert th.agree_cut(97792, 97840, False, lambda v: min(v, 1)) == 97792


# ------------------------------------------------------------------ fakes
N, PAGE, RATIO = 241, 64, 4  # c = 240, page prefix 192, 48 rows
FA = {3: 0, 7: 1}
GDN = {0: 0, 1: 1, 2: 2}
SLOTS = 5


def _pools():
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    g = torch.Generator().manual_seed(18)
    rows = N + 2 * PAGE
    kv = object.__new__(QSATokenToKVPool)
    kv.qsa_compress_ratio = RATIO
    kv.full_attention_layer_id_mapping = dict(FA)
    kv.full_kv_pool = SimpleNamespace(
        k_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in FA],
        v_buffer=[torch.randn(rows, 2, 8, generator=g) for _ in FA],
    )
    kv.qsa_compressed_k_buffer_pool = [torch.randn(rows // RATIO + 1, 1, 4, generator=g) for _ in FA]
    rp = object.__new__(HybridReqToTokenPool)
    rp.mamba_map = dict(GDN)
    rp.mamba_pool = SimpleNamespace(mamba_cache=SimpleNamespace(
        temporal=torch.randn(len(GDN), SLOTS, 2, 4, 4, generator=g),
        conv=[torch.randn(len(GDN), SLOTS, 6, 3, generator=g)],
    ))
    alloc = SimpleNamespace(get_kvcache=lambda: kv)
    return kv, rp, alloc


def _req(ids, end):
    return SimpleNamespace(rid="weg2-0-4", origin_input_ids=ids, full_untruncated_fill_ids=ids, extra_key=None,
                           extend_range=SimpleNamespace(end=end), mamba_pool_idx=torch.tensor(2))


@pytest.fixture
def arena(tmp_path, monkeypatch):
    import sglang.srt.managers.schedule_policy as sp

    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setattr(sp, "_WEG2_END_ANCHOR", True)
    th._CAPTURES.clear()
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True):
        yield tmp_path


def _join_publishers():
    for t in threading.enumerate():
        if t.name == "weg2-tail-publish":
            t.join(10)


def test_capture_only_at_the_chunk_that_ends_at_the_cut(arena):
    kv, rp, alloc = _pools()
    ids = list(range(N))
    assert not th.capture_state(_req(ids, 192), rp, alloc, PAGE, None)  # an earlier chunk
    assert not th.capture_state(_req(ids, N), rp, alloc, PAGE, None)
    assert th.capture_state(_req(ids, 240), rp, alloc, PAGE, None)
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(False):
        th._CAPTURES.clear()
        assert not th.capture_state(_req(ids, 240), rp, alloc, PAGE, None)
        assert not th._CAPTURES


def test_publish_selects_the_state_slot_and_the_partial_page_rows(arena):
    kv, rp, alloc = _pools()
    ids = list(range(N))
    req = _req(ids, 240)
    assert th.capture_state(req, rp, alloc, PAGE, None)
    state_at_c = rp.mamba_pool.mamba_cache.temporal[:, 2].clone()
    rp.mamba_pool.mamba_cache.temporal[:, 2] += 1.0  # the final chunk overwrites the working slot
    kv_indices = torch.arange(N) + PAGE  # token slots, page-aligned base
    th.publish_rows(req, kv_indices, alloc, "pp0-1")
    _join_publishers()
    (h,) = th.headers_for("weg2-0-4")
    assert (h.spec.page_prefix, h.spec.cut, h.spec.rows) == (192, 240, 48)
    assert h.fa_layers == [3, 7] and h.gdn_layers == [0, 1, 2]
    bundle = th.verify_part(h)
    rows = kv_indices[192:240]
    for gid, local in FA.items():
        k, v, c = bundle["fa"][gid]
        assert torch.equal(k, kv.full_kv_pool.k_buffer[local][rows])
        assert torch.equal(v, kv.full_kv_pool.v_buffer[local][rows])
        assert torch.equal(c, kv.qsa_compressed_k_buffer_pool[local][rows[::RATIO] // RATIO])
        assert c.shape[0] == 48 // RATIO
    for gid, local in GDN.items():
        t, conv = bundle["gdn"][gid]
        assert torch.equal(t[0], state_at_c[local])  # the state AFTER c, not after N
        assert torch.equal(conv[0], rp.mamba_pool.mamba_cache.conv[0][local, 2])


def test_tampered_payload_is_refused(arena):
    spec = th.spec_for("weg2-0-4", list(range(N)), None, PAGE, RATIO)
    fa = {3: (torch.ones(48, 2), torch.zeros(48, 2))}
    h = th.write_part(spec, "pp1-9", fa, {5: (torch.ones(1, 3),)})
    assert th.verify_part(h) is not None
    _j, ppath = th.part_paths("weg2-0-4", "pp1-9")
    torch.save({"fa": {3: (torch.ones(48, 2), torch.ones(48, 2))}, "gdn": {5: (torch.ones(1, 3),)}}, ppath)
    assert th.verify_part(h) is None


def test_readiness_needs_every_local_layer_with_its_shape():
    spec = th.spec_for("weg2-0-4", list(range(N)), None, PAGE, RATIO)
    a = th.TailHeader(spec=spec, part="pp0", fa_layers=[3], gdn_layers=[0, 1], fa_row_shapes={"3": [2, 8]},
                      gdn_row_shapes={"0": [5, 2, 4, 4], "1": [5, 2, 4, 4]}, fa_digest="", gdn_digest="", nbytes=0)
    b = th.TailHeader(spec=spec, part="pp1", fa_layers=[7], gdn_layers=[2], fa_row_shapes={"7": [2, 8]},
                      gdn_row_shapes={"2": [5, 2, 4, 4]}, fa_digest="", gdn_digest="", nbytes=0)
    need_fa = {3: [2, 8], 7: [2, 8]}
    need_gdn = {g: [5, 2, 4, 4] for g in (0, 1, 2)}
    assert th.local_readiness(spec, [a, b], need_fa, need_gdn) == ""
    assert th.local_readiness(spec, [a], need_fa, need_gdn) == "fa_layer_missing:7"
    assert th.local_readiness(spec, [a, b], need_fa, {**need_gdn, 9: [5, 2, 4, 4]}) == "gdn_layer_missing:9"
    assert th.local_readiness(spec, [a, b], {3: [4, 8], 7: [2, 8]}, need_gdn).startswith("fa_shape:3")
    other = th.spec_for("weg2-0-4", list(range(N + 4)), None, PAGE, RATIO)
    assert th.local_readiness(other, [a, b], need_fa, need_gdn) == "spec_differs:pp0"
    # a rank without GDN/KV layers (Form-A worker) is ready on any complete spec
    assert th.local_readiness(spec, [a, b], {}, {}) == ""


def test_prune_keeps_the_newest_rids(arena):
    for i in range(4):
        spec = th.spec_for(f"weg2-{i}", list(range(N)), None, PAGE, RATIO)
        th.write_part(spec, "pp0", {}, {})
        os.utime(th.part_paths(f"weg2-{i}", "pp0")[0], (1000 + i, 1000 + i))
    th._prune("weg2-3")
    left = sorted({p.split(".tail.")[0] for p in os.listdir(os.path.join(arena, "handoff"))})
    assert left == ["weg2-2", "weg2-3"]
