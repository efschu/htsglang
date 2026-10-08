"""YaRN x2 before its first boot: the persistent L3 identity names the rope.

The persistent NF L3 (XFS, reused across boots) is keyed by token ids only, so
a store must never hold pages rotated by two ropes. The launcher identity
(L3_IDENTITY.json -> directory digest) already carries the sha of each group's
--json-model-override-args, the config sha and the weight-file stat
fingerprint; the rank identity (L3_RANK_IDENTITY.<group>.json, W165) carries
the effective override sha and the revision; P and D with different ropes are
refused (W57, l3_persist_check_rope).

WHAT THIS ADDS / MUST HOLD.
(1) An override that names a rope (rope_scaling, rope_parameters,
    max_position_embeddings; top level or text_config) puts that rope view,
    canonical, into the identity together with the way the runtime applies it
    (L3_ROPE_APPLY): f37d83617d merges a nested text_config override key by key,
    before it the same string replaced the sub-config -- the same override
    string under the two images is not the same rope, so not the same store.
(2) Every rope difference (factor, original length, max_position_embeddings,
    rope_type, top level vs text_config) is a different directory.
(3) Without a rope in the override the identity is byte-identical: the running
    NF store l3-nextflash-...-fe3b0e1fce (override {"language_model_only":true},
    sha 4d77ae4a3f85a614) keeps its directory. Nothing is deleted.
(4) P and D: one view when equal (the refusal of unequal ones stays W57).
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L  # noqa: E402

LMO = '{"language_model_only":true}'
YARN = ('{"language_model_only":true,"text_config":{"rope_parameters":'
        '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}')

# the running store's L3_IDENTITY.json (read 28.09. from
# /spinning/docker-acceptance/nf/store/l3-nextflash-...-fe3b0e1fce, not written)
RUNNING = {
    "form_kv": "", "generation": "706", "kv_cache_dtype": "fp8_e4m3",
    "model_config_sha1": "cbd3f9cb4153e1af2844294f9d4df4af20f6d475",
    "model_path": "/spinning/llm_stuff/club-3090/models-cache/"
                  "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
    "override_d": "4d77ae4a3f85a614", "override_p": "4d77ae4a3f85a614",
    "profile": "nextflash", "vision": "transient",
    "weights_fp": "3c1c07c454411c8f3712cf05def3a4eed4831445",
}


def _model(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "x"}))
    (d / "model-00001.safetensors").write_bytes(b"\0" * 16)
    return str(d)


def _ident(model, override_p, override_d=None):
    flag = lambda o: "--json-model-override-args '%s'" % o if o else ""  # noqa: E731
    return L.l3_persist_identity(
        model, "nextflash", extra_p=flag(override_p),
        extra_d=flag(override_p if override_d is None else override_d), vision="transient")


def test_the_running_store_keeps_its_directory():
    assert L.l3_persist_dir_name(RUNNING).endswith("-fe3b0e1fce")
    import hashlib
    assert hashlib.sha1(LMO.encode()).hexdigest()[:16] == RUNNING["override_p"]


def test_no_rope_in_the_override_no_new_key(tmp_path):
    m = _model(tmp_path)
    for ovr in ("", LMO, '{"x":1}'):
        ident = _ident(m, ovr)
        assert "rope" not in ident and "rope_apply" not in ident, ovr
        assert set(ident) == set(RUNNING)


def test_a_yarn_override_names_the_rope_and_the_apply_semantics(tmp_path):
    m = _model(tmp_path)
    ident = _ident(m, YARN)
    assert ident["rope_apply"] == L.L3_ROPE_APPLY == "merge-v1"
    view = json.loads(ident["rope"])
    assert view == {"text_config": {"rope_parameters": {
        "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 262144}}}
    # the same override string without the apply marker (the pre-f37d
    # identity) is a different directory: its pages were never these
    before = {k: v for k, v in ident.items() if k not in ("rope", "rope_apply")}
    assert L.l3_persist_dir_name(ident) != L.l3_persist_dir_name(before)
    assert L.l3_persist_dir_name(ident) != L.l3_persist_dir_name(_ident(m, LMO))


def test_every_rope_difference_is_a_new_directory(tmp_path):
    m = _model(tmp_path)
    variants = [
        YARN,
        YARN.replace('"factor":2.0', '"factor":4.0'),
        YARN.replace("262144}", "131072}"),
        YARN.replace('"yarn"', '"linear"'),
        '{"language_model_only":true,"rope_scaling":{"rope_type":"yarn","factor":2.0,'
        '"original_max_position_embeddings":262144}}',
        '{"language_model_only":true,"text_config":{"max_position_embeddings":524288}}',
    ]
    names = {L.l3_persist_dir_name(_ident(m, v)) for v in variants}
    assert len(names) == len(variants)
    # key order / whitespace of the SAME rope: one rope view (the override sha
    # still separates the raw strings, conservatively)
    a = _ident(m, YARN)["rope"]
    b = _ident(m, json.dumps(json.loads(YARN), indent=1, sort_keys=False))["rope"]
    assert a == b


def test_p_and_d_share_one_view(tmp_path):
    m = _model(tmp_path)
    same = _ident(m, YARN, YARN)
    assert not same["rope"].startswith("P=")
    diff = _ident(m, LMO, YARN)
    assert diff["rope"].startswith("P=,D=")  # (and W57 refuses it at the launch)
