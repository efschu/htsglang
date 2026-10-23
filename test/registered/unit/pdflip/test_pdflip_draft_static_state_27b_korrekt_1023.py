"""27b-korrekt-1023: the drafter's buffers ride the sleep beside the target's.

MEASURED (27B INT4-AutoRound, D = TP2 on 2x RTX 3080, DFlash2-W8 draft):

* boot dkr27bint4arbar1int4ar10231552 (rc35i4ar2, profile with ``--context-length=131072``):
  D ``SPEC-ACCEPT-PROFILE`` median accept_len 1.03 (242/260 requests < 1.3, the 22-61 token
  knowledge probes included), DEGEN-SUSPECT loops ("the the the", " — — —") and foreign tokens
  ("_ AutoBatchSetEach", "1221 Tats Auto") in long sampled agent answers. Boots 10231400 /
  10231444 (same model, context 262144): median 3.82 / 3.85, 0 loops.
* A/B on the same code 216204f8f7: ab0 (context 262144) accept 3.5-4.9; abctx (131072, trim 0)
  1.03; abratio (131072, --rank-tp-ratio pinned to the healthy 2185,2158) 1.02.

ROOT. The sleep stashed the buffers of the TARGET model alone
(``export_static_state_host(self.tp_worker.model_runner.model)``) and the wake imported them
back; the drafter's one buffer, the RoPE ``cos_sin_cache`` (``PDFLIP-XCHG-COVER
tag=weights_draft buffers=1 buffers_mib=128.0``), sits in ``weights_draft``, whose bytes the
exchange carries PARAMETER by parameter -- so after every wake it was undefined. W84 booked it
as covered (``uncovered=0``) because it assumed every registered buffer comes back by the
stash. Latent while ``reserve_rope_cache_for_long_sequences`` re-allocates the cache after the
load (reserve 262528 > 262144 rows: a new tensor outside the weights region); a context below
the checkpoint's max_position_embeddings (131072 -> reserve 131456) keeps the load-time buffer
inside the tag, and the draft read garbage cos/sin from the first wake on.
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from flliper.srt.constants import GPU_MEMORY_TYPE_WEIGHTS_DRAFT  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from flliper.srt.pdflip import sleep_staging as ss  # noqa: E402
from flliper.srt.pdflip import weight_exchange as wx  # noqa: E402


def _rope_rows(n=64, d=16):
    t = torch.arange(n, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    f = t * inv
    return torch.cat((f.cos(), f.sin()), dim=-1)


class _Rope(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("cos_sin_cache", rows.clone(), persistent=False)


class _Draft(nn.Module):
    """The DFlash2 draft's shape that matters here: one parameter, and one rope
    cache shared by the layers (get_rope's cache dict) -> ONE registered buffer."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Parameter(torch.zeros(8, 8), requires_grad=False)
        rope = _Rope(_rope_rows())
        self.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.layers:
            layer.self_attn = nn.Module()
            layer.self_attn.rotary_emb = rope


class _Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        self.rope = _Rope(_rope_rows(32, 8))


def _owner(draft, target, via="draft_worker"):
    tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=target))
    runner = types.SimpleNamespace(model=draft)
    if via == "draft_worker":
        return types.SimpleNamespace(tp_worker=tp_worker,
                                     draft_worker=types.SimpleNamespace(draft_model_runner=runner))
    # group P's last stage: no draft_worker, the drafter sits in the draft-KV producer
    return types.SimpleNamespace(tp_worker=tp_worker, draft_worker=None,
                                 draft_kv_producer=types.SimpleNamespace(draft_runner=runner))


def _simulate_pause(model):
    """What a paused-and-remapped tag leaves in a buffer nobody refills."""
    for _n, b in model.named_buffers():
        b.fill_(float("nan"))


def test_switch_default_on():
    assert ss.draft_static_state_on({})
    for off in ("0", "false", "no", "off"):
        assert not ss.draft_static_state_on({ss.ENV_DRAFT_STATIC: off})


def test_draft_rope_cache_survives_a_sleep_both_stash_forms(monkeypatch):
    for host in (True, False):
        monkeypatch.delenv(ss.ENV_DRAFT_STATIC, raising=False)
        draft, target = _Draft(), _Target()
        want = _rope_rows()
        stash = ss.export_draft_static_state(draft, target, host=host)
        assert stash is not None
        assert ss.LAST_DRAFT_EXPORT["n"] == 1  # one shared cache, one storage
        assert ss.LAST_DRAFT_EXPORT["names"] == ["layers.0.self_attn.rotary_emb.cos_sin_cache"]
        _simulate_pause(draft)
        assert torch.isnan(draft.layers[1].self_attn.rotary_emb.cos_sin_cache).all()
        assert ss.import_draft_static_state(draft, stash) == 1
        assert torch.equal(draft.layers[1].self_attn.rotary_emb.cos_sin_cache, want)


def test_the_target_stash_numbers_stay_the_targets(monkeypatch):
    monkeypatch.delenv(ss.ENV_DRAFT_STATIC, raising=False)
    ss.export_static_state_host(_Target())
    before = dict(ss.LAST_EXPORT)
    ss.export_draft_static_state(_Draft(), _Target(), host=True)
    assert ss.LAST_EXPORT == before


def test_shared_with_target_meta_and_same_model_are_not_carried():
    target = _Target()
    draft = _Draft()
    draft.shared = _Rope(torch.zeros(1))
    draft.shared.cos_sin_cache = target.rope.cos_sin_cache  # the target's storage
    draft.ghost = _Rope(torch.zeros(1))
    draft.ghost.cos_sin_cache = torch.empty(4, device="meta")
    names = [n for n, _b in ss.draft_static_buffers(draft, target)]
    assert names == ["layers.0.self_attn.rotary_emb.cos_sin_cache"]
    assert ss.draft_static_buffers(target, target) == []
    assert ss.draft_static_buffers(None, target) == []


def test_switch_off_carries_nothing(monkeypatch):
    monkeypatch.setenv(ss.ENV_DRAFT_STATIC, "0")
    assert ss.export_draft_static_state(_Draft(), _Target()) is None
    assert ss.import_draft_static_state(_Draft(), None) == 0


def test_weight_updater_helpers_find_the_drafter_on_D_and_on_P(monkeypatch):
    monkeypatch.delenv(ss.ENV_DRAFT_STATIC, raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_GROUP", raising=False)
    for via in ("draft_worker", "producer"):
        draft, target = _Draft(), _Target()
        owner = _owner(draft, target, via=via)
        stash = wu._pdflip_export_draft_static_state(owner)
        assert stash is not None, via
        _simulate_pause(draft)
        _simulate_pause(target)  # the target's own stash is not this helper's job
        assert wu._pdflip_import_draft_static_state(owner, stash) == 1
        assert torch.equal(draft.layers[0].self_attn.rotary_emb.cos_sin_cache, _rope_rows())
        assert torch.isnan(target.rope.cos_sin_cache).all()


def test_sleep_and_wake_are_wired():
    """The sleep exports the drafter's stash BEFORE the family pause, right after the
    target's; the wake imports it right after the target's import."""
    src = inspect.getsource(wu)
    i_t = src.index("_ss.export_static_state_host(self.tp_worker.model_runner.model)")
    i_d = src.index("self.stashed_draft_static_state = _pdflip_export_draft_static_state(self)")
    i_barrier = src.index('_pdflip_ph("static_export")')
    assert i_t < i_d < i_barrier
    j_t = src.index("_import_static_state(\n                    self.tp_worker.model_runner.model,")
    j_d = src.index("_pdflip_import_draft_static_state(self, self.stashed_draft_static_state)")
    j_sub = src.index('_pdflip_sub_t("static_import", _sub_t0)', j_t)
    assert j_t < j_d < j_sub


class _Exchange:
    def __enter__(self):
        self.saved = os.environ.get(wx.WEIGHT_SOURCE_ENV)
        os.environ[wx.WEIGHT_SOURCE_ENV] = wx.WEIGHT_SOURCE_EXCHANGE
        return self

    def __exit__(self, *_a):
        if self.saved is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = self.saved
        return False


def _draft_rows():
    model = _Draft()
    planned = {}
    for name, p in model.named_parameters():
        tag = wx.tag_of_parameter_name(name, region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
        planned.setdefault(tag, {})[name] = int(p.untyped_storage().nbytes())
    return wx.build_coverage(model, rank=0, planned_bytes_by_tag=planned,
                             tag_bytes=lambda _t: 0, region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT)


def test_w84_books_the_draft_buffer_by_what_really_restores_it(monkeypatch):
    with _Exchange():
        monkeypatch.delenv(ss.ENV_DRAFT_STATIC, raising=False)
        row = _draft_rows()[GPU_MEMORY_TYPE_WEIGHTS_DRAFT]
        assert row.ok and row.uncovered == ()
        assert row.buffers_bytes == _rope_rows().numel() * 4
        monkeypatch.setenv(ss.ENV_DRAFT_STATIC, "0")
        row = _draft_rows()[GPU_MEMORY_TYPE_WEIGHTS_DRAFT]
        assert not row.ok
        assert [t.name for t in row.uncovered] == ["layers.0.self_attn.rotary_emb.cos_sin_cache"]


def test_the_targets_buffers_stay_covered_whatever_the_draft_switch(monkeypatch):
    monkeypatch.setenv(ss.ENV_DRAFT_STATIC, "0")
    assert wx.buffers_restored_by_static_state("weights")
    assert wx.buffers_restored_by_static_state("weights_0")
    assert not wx.buffers_restored_by_static_state(GPU_MEMORY_TYPE_WEIGHTS_DRAFT)


def test_the_trigger_reserve_below_the_boot_length_keeps_the_load_time_buffer():
    """Why only --context-length 131072 showed it: the reserve re-allocates the cache
    (a new tensor, born after the weights region closed) only when it exceeds the
    load-time length; below it the module keeps reading the buffer born in the tag."""
    from flliper.srt.layers.rotary_embedding.base import RotaryEmbedding
    from flliper.srt.utils.common import reserve_rope_cache_for_long_sequences

    def fresh():
        r = RotaryEmbedding.__new__(RotaryEmbedding)
        nn.Module.__init__(r)
        r.rotary_dim = 16
        r.max_position_embeddings = 262144
        r.base = 10000
        r._lazy_cos_sin = None
        r.register_buffer("cos_sin_cache", torch.zeros(262144, 16), persistent=False)
        r._cos_sin_cache_inv_freq = lambda: 1.0 / (
            10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        holder = nn.Module()
        holder.rope = r
        return holder

    for ctx, moved in ((131072, False), (262144, True)):
        m = fresh()
        ptr = m.rope.cos_sin_cache.data_ptr()
        sa = types.SimpleNamespace(context_length=ctx, speculative_num_steps=1,
                                   speculative_num_draft_tokens=8)
        mc = types.SimpleNamespace(context_len=ctx, hf_text_config=types.SimpleNamespace())
        reserve_rope_cache_for_long_sequences(m, sa, mc, None)
        assert (m.rope.cos_sin_cache.data_ptr() != ptr) is moved, ctx
