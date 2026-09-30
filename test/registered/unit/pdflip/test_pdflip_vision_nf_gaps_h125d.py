"""H125d (NF vision in the release): the desk gaps of tmp/vision_nf/PLAN.md.

  2a  Qwen4-Exp merges image rows into the INPUT embeddings only: a
      checkpoint with a non-empty ``deepstack_visual_indexes`` is refused at
      construction, a deepstack tensor that reaches the language model is
      refused there -- never dropped silently. The NF checkpoint (Minachist)
      has ``[]`` and passes.
  4a  PLE at image positions: the language model feeds PLE (and so the
      n-gram history it commits, which the decode and D's hand-off read) the
      image_token_id ids, not the hash pad ids -- also with the image at the
      prompt END, where the history IS the image.
  5a  P/D key identity: an image's pad value (radix/L3 key) is a content
      hash, equal across processes and different for a different image;
      ``FLLIPER_MM_SKIP_COMPUTE_HASH`` (a uuid per process) is refused by name
      under ``--pdflip-vision transient``.
  7   ``FLLIPER_PDFLIP_ENABLE_MROPE_IMAGE_EXTENT_ONLY``: decode and text extends
      take mrope row 0 through the 1D rotary; only an extend with image inputs
      takes the 3D path; the model's flag is restored on every exit.

Hermetic, CPU, no server.
"""

import json
import os
import subprocess
import sys
import types

import pytest
import torch

from flliper.srt.model_executor.forward_batch_info import ForwardMode
from flliper.srt.models import qwen4_exp as q4

NF_CKPT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
IMAGE_TOKEN_ID = 248056  # the NF checkpoint's image_token_id


def _cfg(indexes):
    return types.SimpleNamespace(vision_config=types.SimpleNamespace(deepstack_visual_indexes=indexes))


# --------------------------------------------------------------------- 2a --


def test_an_empty_deepstack_list_passes():
    q4.refuse_deepstack_config(_cfg([]))
    q4.refuse_deepstack_config(_cfg(None))
    q4.refuse_deepstack_config(types.SimpleNamespace())  # no vision config at all


def test_a_deepstack_checkpoint_is_refused_by_name():
    with pytest.raises(q4.Qwen4ExpDeepstackRefused, match="H125d QWEN4EXP DEEPSTACK REFUSED"):
        q4.refuse_deepstack_config(_cfg([8, 16, 24]))


@pytest.mark.skipif(not os.path.exists(os.path.join(NF_CKPT, "config.json")), reason="NF checkpoint absent")
def test_the_nf_checkpoint_has_no_deepstack():
    with open(os.path.join(NF_CKPT, "config.json")) as f:
        cfg = json.load(f)
    vc = cfg["vision_config"]
    assert vc["deepstack_visual_indexes"] == []
    assert cfg["image_token_id"] == IMAGE_TOKEN_ID
    q4.refuse_deepstack_config(_cfg(vc["deepstack_visual_indexes"]))


def test_the_constructor_refuses_before_building_anything():
    import inspect

    src = inspect.getsource(q4.Qwen4ExpForConditionalGeneration.__init__)
    assert src.index("refuse_deepstack_config(config)") < src.index("super().__init__(")


def _vl_model():
    m = q4.Qwen4ExpVLModel.__new__(q4.Qwen4ExpVLModel)
    torch.nn.Module.__init__(m)
    m.last_hc_hidden_states = None
    m.ple_input_ids = None
    return m


def test_a_deepstack_tensor_reaching_the_language_model_is_refused(monkeypatch):
    called = []
    monkeypatch.setattr(q4.Qwen4ExpModel, "forward", lambda self, **kw: called.append(kw) or torch.zeros(1))
    m = _vl_model()
    fb = types.SimpleNamespace(input_ids=torch.tensor([1, 2]))
    with pytest.raises(q4.Qwen4ExpDeepstackRefused, match="input_deepstack_embeds"):
        m.forward(None, torch.arange(2), fb, input_embeds=torch.zeros(2, 4),
                  input_deepstack_embeds=torch.zeros(2, 8))
    assert called == []
    m.forward(None, torch.arange(2), fb, input_embeds=torch.zeros(2, 4))
    assert len(called) == 1


# --------------------------------------------------------------------- 4a --


def _pads(n, seed=12345):
    from flliper.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    return [MM_PAD_SHIFT_VALUE + seed + i * 0 for i in range(n)]


def test_ple_sees_image_token_ids_with_the_image_at_the_prompt_end(monkeypatch):
    """The last n-gram history an extend commits is its last ids: with the
    image at the end that history is image positions -- it must be
    image_token_id there, or D's first decode token hashes pad ids."""
    seen = {}
    monkeypatch.setattr(q4.Qwen4ExpModel, "forward",
                        lambda self, **kw: seen.setdefault("ids", kw["input_ids"]) is None or torch.zeros(1))
    ids = torch.tensor([11, 12, 13] + _pads(6))
    m = _vl_model()
    m.ple_input_ids = q4.ple_ids_for_images(ids, IMAGE_TOKEN_ID)
    fb = types.SimpleNamespace(input_ids=ids)
    m.forward(None, torch.arange(len(ids)), fb, input_embeds=torch.zeros(len(ids), 4))
    got = seen["ids"].tolist()
    assert got[:3] == [11, 12, 13]
    assert got[3:] == [IMAGE_TOKEN_ID] * 6  # the tail = the committed history
    assert max(got) < 1_000_000  # no hash pad id reaches PLE


def _cg_model(*, ple_image_token_id=IMAGE_TOKEN_ID, mrope_on=True, image_extent_only=False):
    m = q4.Qwen4ExpForConditionalGeneration.__new__(q4.Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(m)
    m.ple_image_token_id = ple_image_token_id
    m.is_mrope_enabled = mrope_on
    m.mrope_image_extent_only = image_extent_only
    m.model = types.SimpleNamespace(ple_input_ids=None, last_hc_hidden_states=None,
                                    start_layer=0, end_layer=1)
    return m


class _FB:
    def __init__(self, mode, ids, images=False, mrope=None):
        self.forward_mode = mode
        self.input_ids = ids
        self._images = images
        self.mrope_positions = mrope

    def contains_mm_inputs(self):
        return self._images

    def contains_image_inputs(self):
        return self._images


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def fake(self, input_ids, positions, forward_batch, get_embedding=False, pp_proxy_tensors=None):
        calls.append(dict(ple=None if self.model.ple_input_ids is None else self.model.ple_input_ids.clone(),
                          positions=positions, mrope=self.is_mrope_enabled))
        return torch.zeros(1)

    from flliper.srt.models import qwen3_vl

    monkeypatch.setattr(qwen3_vl.Qwen3VLForConditionalGeneration, "forward", fake)
    return calls


def test_an_image_extend_hands_ple_the_mapped_ids_on_every_group(capture):
    """P and D both tokenize images under transient, so both map; the ids are
    cleared after the forward (never leak into the next batch)."""
    ids = torch.tensor([5] + _pads(3) + [6])
    m = _cg_model()
    m.forward(ids, torch.arange(5), _FB(ForwardMode.EXTEND, ids, images=True))
    assert capture[-1]["ple"].tolist() == [5] + [IMAGE_TOKEN_ID] * 3 + [6]
    assert m.model.ple_input_ids is None
    m.forward(ids[:1], torch.arange(1), _FB(ForwardMode.DECODE, ids[:1], images=True))
    assert capture[-1]["ple"] is None  # decode: one real token, history from the pool


# --------------------------------------------------------------------- 5a --

_HASH_PROG = r"""
import sys, torch
from flliper.srt.managers.schedule_batch import MultimodalDataItem, Modality
torch.manual_seed(int(sys.argv[1]))
px = torch.rand(4, 3 * 2 * 16 * 16)
it = MultimodalDataItem(modality=Modality.IMAGE, feature=px)
it.set_pad_value()
print(it.pad_value)
"""


def _pad_in_fresh_process(seed, env_extra=None):
    env = dict(os.environ)
    env.pop("FLLIPER_MM_SKIP_COMPUTE_HASH", None)
    env.update(env_extra or {})
    out = subprocess.run([sys.executable, "-c", _HASH_PROG, str(seed)], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return int(out.stdout.strip().splitlines()[-1])


def test_the_image_key_is_a_content_hash_stable_across_processes():
    """P's and D's tokenizers are different processes: the same pixels must
    give the same pad value (radix/L3 key), a different image a different one."""
    from flliper.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    a1, a2, b = _pad_in_fresh_process(1), _pad_in_fresh_process(1), _pad_in_fresh_process(2)
    assert a1 == a2
    assert a1 != b
    assert a1 >= MM_PAD_SHIFT_VALUE and b >= MM_PAD_SHIFT_VALUE


def test_skip_hash_makes_the_key_per_process_which_is_why_it_is_refused():
    on = {"FLLIPER_MM_SKIP_COMPUTE_HASH": "1"}
    assert _pad_in_fresh_process(1, on) != _pad_in_fresh_process(1, on)


def _env(group, vision):
    from flliper.srt.pdflip import launcher as lz

    return lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group=group, vision=vision)


def test_transient_refuses_skip_hash_on_both_groups(monkeypatch):
    from flliper.srt.pdflip import launcher as lz

    monkeypatch.setenv(lz.MM_SKIP_HASH_ENV, "1")
    for group in ("P", "D"):
        with pytest.raises(lz.PdFlipLaunchRefused, match="W111.*FLLIPER_MM_SKIP_COMPUTE_HASH"):
            _env(group, lz.VISION_TRANSIENT)
        _env(group, lz.VISION_OFF)  # text-only: no image key, the operator's business
    monkeypatch.setenv(lz.MM_SKIP_HASH_ENV, "0")
    _env("P", lz.VISION_TRANSIENT)


# ---------------------------------------------------------------------- 7 --


def _mrope(n, delta=0):
    base = torch.arange(n) + delta
    return torch.stack([base, base, base])


def test_switch_off_keeps_mrope_on_every_batch(capture):
    m = _cg_model(image_extent_only=False)
    ids = torch.tensor([1])
    pos = torch.tensor([9])
    m.forward(ids, pos, _FB(ForwardMode.DECODE, ids, mrope=_mrope(1, 40)))
    assert capture[-1]["mrope"] is True and capture[-1]["positions"] is pos


def test_decode_takes_row_zero_on_the_1d_rotary_and_restores_the_flag(capture):
    m = _cg_model(image_extent_only=True)
    ids = torch.tensor([1, 2])
    mp = _mrope(2, delta=37)  # an image request's decode: row 0 carries the delta
    m.forward(ids, torch.tensor([100, 200]), _FB(ForwardMode.DECODE, ids, images=True, mrope=mp))
    c = capture[-1]
    assert c["mrope"] is False
    assert c["positions"].tolist() == mp[0].tolist()
    assert m.is_mrope_enabled is True


def test_a_text_extend_takes_row_zero(capture):
    m = _cg_model(image_extent_only=True)
    ids = torch.arange(4)
    mp = _mrope(4)
    m.forward(ids, torch.arange(4), _FB(ForwardMode.EXTEND, ids, images=False, mrope=mp))
    assert capture[-1]["mrope"] is False and capture[-1]["positions"].tolist() == [0, 1, 2, 3]


def test_an_image_extend_keeps_the_3d_path(capture):
    m = _cg_model(image_extent_only=True)
    ids = torch.tensor([5] + _pads(3))
    pos = torch.arange(4)
    m.forward(ids, pos, _FB(ForwardMode.EXTEND, ids, images=True, mrope=_mrope(4)))
    assert capture[-1]["mrope"] is True and capture[-1]["positions"] is pos


def test_the_flag_is_restored_when_the_forward_raises(monkeypatch):
    from flliper.srt.models import qwen3_vl

    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(qwen3_vl.Qwen3VLForConditionalGeneration, "forward", boom)
    m = _cg_model(image_extent_only=True)
    ids = torch.tensor([1])
    with pytest.raises(RuntimeError, match="boom"):
        m.forward(ids, torch.tensor([3]), _FB(ForwardMode.DECODE, ids, mrope=_mrope(1)))
    assert m.is_mrope_enabled is True


def test_mrope_for_batch_is_decided_by_mode_and_image_inputs():
    assert q4.mrope_for_batch(_FB(ForwardMode.EXTEND, None, images=True)) is True
    assert q4.mrope_for_batch(_FB(ForwardMode.EXTEND, None, images=False)) is False
    assert q4.mrope_for_batch(_FB(ForwardMode.DECODE, None, images=True)) is False
    assert q4.mrope_for_batch(_FB(ForwardMode.IDLE, None, images=True)) is False
    assert q4.mrope_for_batch(_FB(ForwardMode.TARGET_VERIFY, None, images=True)) is False


def test_the_switch_is_registered_and_off_by_default():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_ENABLE_MROPE_IMAGE_EXTENT_ONLY.get() is False
