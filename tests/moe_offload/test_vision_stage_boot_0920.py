# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 10 -- ARMING the transient vision stage as part of the boot.

HERMETIC: no CUDA, no GPU, no NVML, no network, no checkpoint data (a
synthetic safetensors header in ``tmp_path`` carries the tower).  Every side
effect the arming has -- the NVML read, the CUDA-context probe child, the
tower module build -- is injected, which is the same discipline
``StageHooks`` follows and for the same reason: the whole path has to be
drivable at a desk or it is only ever tested on metal.

THE DEFECT THIS SLICE CLOSES, restated because it is the point of every test
below: slice 9 built the service and the seam, and NOTHING CALLED ``install``.
A boot with ``--weg2-vision transient`` was therefore byte-for-byte a boot
without it -- the seam a permanent no-op, the items leaving with ``feature``
and no rows, and the failure surfacing three hops later inside a rank.  A
silent no-op is the one outcome this path may not have.

WHAT IS PINNED HERE

1. **The hook set is complete** against the design's §6 table, and
   ``pause_tag``/``resume_tag`` being ``None`` counts as PRESENT -- ``None``
   is their correct value in this process, and a truthy check would have
   called the honest default a missing hook.
2. **``ctx_bytes`` is a MEASURED post**, from a child process that dies, and
   the measurement is pinned by the card's NVML UUID rather than an ordinal.
   A probe that measures nothing REFUSES instead of planning a free context.
3. **No service under ``transient`` refuses BY NAME**, at boot (W111) and
   again on the first image request (W112) quoting the boot reason -- and the
   processor seam does NOT swallow that one.
4. **W103 (video) is untouched** in every mode, ``transient`` included.
5. **The text path is unchanged without the flag**: the argv, the published
   environment, the arming call and the seam all no-op.
"""

import json
import logging
import struct
import types

import pytest

from sglang.srt.planner import vision_stage as vs
from sglang.srt.planner import vision_stage_load as vsl
from sglang.srt.weg2 import front as fr
from sglang.srt.weg2 import launcher as lz
from sglang.srt.weg2 import vision_stage_boot as vsb
from sglang.srt.weg2 import vision_stage_service as vss


# --------------------------------------------------------------- fakes --


class _Dev:
    def __init__(self, index, uuid, total_mib):
        self.index = index
        self.uuid = uuid
        self.total_mib = total_mib


class _Mem:
    def __init__(self, free_mib, total_mib):
        self.free_mib = free_mib
        self.total_mib = total_mib


def _snapshot(free=(2100, 700, 2140)):
    """Three cards, the rig's shape: 3080 / 5090 / 3080, NVML order."""
    totals = (20480, 32607, 20480)
    return [
        (_Dev(i, f"GPU-fake-{i}", totals[i]), _Mem(free[i], totals[i]))
        for i in range(3)
    ]


CTX_MIB = 420


def _probe_ok(argv, env):
    """A child that measured a 420 MiB context on whatever UUID it was given."""
    uuid = env["CUDA_VISIBLE_DEVICES"]
    before = 2100 * (1 << 20)
    return (
        "some torch chatter\n"
        + vsb._PROBE_MARKER
        + json.dumps(
            {
                "uuid": uuid,
                "free_before_bytes": before,
                "free_after_bytes": before - CTX_MIB * (1 << 20),
                "ctx_bytes": CTX_MIB * (1 << 20),
            }
        )
        + "\n"
    )


TOWER_BYTES = 921_460_192
TOWER_PIECES = 4


@pytest.fixture
def model_dir(tmp_path):
    """A checkpoint whose tower is one contiguous extent in one shard."""
    names = [f"model.visual.blocks.{i}.attn.qkv.weight" for i in range(TOWER_PIECES)]
    per = TOWER_BYTES // TOWER_PIECES
    header, off = {}, 1024  # text weights before the tower
    header["model.embed_tokens.weight"] = {
        "dtype": "BF16", "shape": [8, 8], "data_offsets": [0, off]
    }
    for n in names:
        header[n] = {"dtype": "BF16", "shape": [per // 2], "data_offsets": [off, off + per]}
        off += per
    blob = json.dumps(header).encode()
    shard = tmp_path / "model-00001-of-00001.safetensors"
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        # The DATA is never read by this slice: only the header is parsed.
        fh.truncate(8 + len(blob) + off)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in header}})
    )
    return str(tmp_path)


def _vision_config():
    """The serving checkpoint's vision_config (design §1.4)."""
    return types.SimpleNamespace(
        depth=27,
        hidden_size=1152,
        intermediate_size=4304,
        num_heads=16,
        out_hidden_size=5120,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        deepstack_visual_indexes=[],
    )


def _hf_config():
    return types.SimpleNamespace(vision_config=_vision_config(), rms_norm_eps=1e-6)


@pytest.fixture(autouse=True)
def _clean_service():
    vss.reset_for_test()
    yield
    vss.reset_for_test()


def _build(model_dir, **kw):
    kw.setdefault("snapshot", _snapshot)
    kw.setdefault("context_runner", _probe_ok)
    return vsb.build_service(model_dir=model_dir, hf_config=_hf_config(), **kw)


# ------------------------------------------------ 1. the hook set is whole --


def test_every_hook_the_design_names_is_present_on_the_built_service(model_dir):
    """The §6 table IS the contract; a partial service fails at a seam."""
    service, _ctx, _cfg = _build(model_dir)
    assert vsb.missing_hooks(service) == ()
    for hook in vsb.REQUIRED_HOOKS:
        assert hasattr(service, hook), hook


def test_the_hook_list_is_exactly_the_designs_table():
    assert set(vsb.REQUIRED_HOOKS) == {
        "nvml_snapshot", "h2d_gbps", "tower", "load_tower", "encode",
        "release_tower", "read_gbps", "pause_tag", "resume_tag",
    }


def test_pause_and_resume_are_None_and_that_is_PRESENT_not_missing(model_dir):
    """The naive-truthy shape would have called the honest default a gap.

    `None` is the correct value here -- band displacement lives in the rank
    processes -- and `eviction_available` reads it as "forbidden", which is
    what makes a band-needing placement refuse by name instead of being
    silently attempted.
    """
    service, _ctx, _cfg = _build(model_dir)
    assert service.pause_tag is None and service.resume_tag is None
    assert vsb.missing_hooks(service) == ()
    assert service.eviction_available is False


def test_a_service_that_lost_a_hook_is_reported_as_missing_it():
    """`missing_hooks` must be able to say no, or its yes means nothing."""
    stub = types.SimpleNamespace(nvml_snapshot=None, h2d_gbps={}, tower=None)
    assert set(vsb.missing_hooks(stub)) == {
        "load_tower", "encode", "release_tower", "read_gbps",
        "pause_tag", "resume_tag",
    }


def test_the_read_rate_handed_to_the_planner_is_the_BUFFERED_one(model_dir):
    """3.85 GB/s is the O_DIRECT path and it is NOT built (slice 7).

    Planning 239 ms and getting 853 ms is the trap this constant exists for.
    """
    service, _ctx, _cfg = _build(model_dir)
    assert service.read_gbps == vsl.MEASURED_LOADER_GBPS == 1.08
    assert vsl.LOADER_IS_BUFFERED is True
    assert service.read_gbps != vs.MEASURED_ODIRECT_GBPS


def test_the_link_rates_are_the_measured_ones_per_card(model_dir):
    """Memory RANG-LINK-ZUORDNUNG: rank 1 is the x4 slot, not a nominal 16."""
    service, _ctx, _cfg = _build(model_dir)
    assert service.h2d_gbps == {0: 14.4, 1: 6.5, 2: 13.3}


def test_the_encoder_config_comes_from_the_CHECKPOINT_not_the_defaults(model_dir):
    """The defaults ARE this checkpoint's numbers, which is why they may not
    be relied on: a right-by-luck default is indistinguishable from a stale one."""
    hf = _hf_config()
    hf.vision_config.out_hidden_size = 4096
    service, _ctx, cfg = vsb.build_service(
        model_dir=model_dir, hf_config=hf,
        snapshot=_snapshot, context_runner=_probe_ok,
    )
    assert cfg.out_hidden_size == 4096
    assert service.encoder_config.out_hidden_size == 4096
    assert vs.VisionEncoderConfig().out_hidden_size == 5120  # the default differs


def test_a_checkpoint_with_no_vision_config_refuses_by_name():
    with pytest.raises(vsb.VisionStageArmRefused) as e:
        vsb.encoder_config_from_hf(None)
    assert "vision_config" in str(e.value)


def test_a_vision_config_missing_a_field_refuses_instead_of_defaulting():
    cfg = _vision_config()
    del cfg.intermediate_size
    with pytest.raises(vsb.VisionStageArmRefused) as e:
        vsb.encoder_config_from_hf(cfg)
    assert "intermediate_size" in str(e.value)


# ------------------------------------------- 2. ctx_bytes is a MEASURED post --


def test_the_measured_context_lands_in_the_tower_spec_as_its_own_post(model_dir):
    """Not folded into the weights, not estimated: its own line in `posts`."""
    service, ctx, _cfg = _build(model_dir)
    assert ctx.ctx_bytes == CTX_MIB * (1 << 20)
    assert service.tower.ctx_bytes == ctx.ctx_bytes
    assert dict(service.tower.posts)["cuda context"] == ctx.ctx_bytes
    assert service.tower.weight_bytes == TOWER_BYTES
    assert service.tower.total_bytes == (
        service.tower.weight_bytes
        + service.tower.ctx_bytes
        + service.tower.activation_bytes
        + service.tower.embedding_bytes
    )


def test_the_probe_pins_the_card_by_UUID_and_never_by_ordinal():
    """Torch's enumeration and NVML's can diverge; the UUID cannot."""
    argv, env = vsb.context_probe_command("GPU-abc")
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-abc"
    assert argv[1] == "-c"
    assert "GPU-abc" in argv
    assert "cuda:0" in argv[2]  # inside the child, card 0 IS that UUID


def test_the_probe_drops_a_preload_it_must_not_inherit(monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/some/tms.so")
    _argv, env = vsb.context_probe_command("GPU-abc")
    assert "LD_PRELOAD" not in env


def test_a_probe_without_a_uuid_refuses():
    with pytest.raises(vsb.VisionStageArmRefused):
        vsb.context_probe_command("")


def test_both_free_readings_are_kept_not_only_the_delta(model_dir):
    """A 420 MiB delta out of 2 GiB free and out of 30 GiB free are different
    facts; the line prints the denominator."""
    _service, ctx, _cfg = _build(model_dir)
    line = ctx.log_line()
    assert "free_before=" in line and "free_after=" in line and "ctx_bytes=" in line
    assert "card=0" in line


def test_a_context_that_does_not_show_up_in_NVML_REFUSES(model_dir):
    """Zero-cost context = the probe measured something else. Refuse, never plan."""
    def zero(argv, env):
        return vsb._PROBE_MARKER + json.dumps({
            "uuid": env["CUDA_VISIBLE_DEVICES"],
            "free_before_bytes": 100, "free_after_bytes": 100, "ctx_bytes": 0,
        })

    with pytest.raises(vsb.VisionStageArmRefused) as e:
        _build(model_dir, context_runner=zero)
    assert "ctx_bytes=0" in str(e.value)


def test_a_silent_probe_refuses_and_quotes_what_it_actually_said(model_dir):
    def silent(argv, env):
        return "ImportError: no torch here\n"

    with pytest.raises(vsb.VisionStageArmRefused) as e:
        _build(model_dir, context_runner=silent)
    assert "no torch here" in str(e.value)


def test_the_probe_card_is_deterministic_and_not_the_emptiest():
    """Picking by free memory makes the arming depend on what else ran."""
    idx, uuid = vsb.probe_card(_snapshot(free=(10, 30000, 20)), vsb.MEASURED_H2D_GBPS)
    assert (idx, uuid) == (0, "GPU-fake-0")


def test_a_card_with_no_measured_link_rate_cannot_host_the_probe():
    with pytest.raises(vsb.VisionStageArmRefused) as e:
        vsb.probe_card(_snapshot(), {7: 1.0})
    assert "measured h2d rate" in str(e.value)


# ---------------------------------------- the two geometry-derived posts --


def test_the_activation_is_ONE_block_not_the_sum_over_depth():
    """Summing 27 blocks books 27x the truth and refuses cards that fit."""
    cfg = vs.VisionEncoderConfig()
    rows = cfg.patch_rows(1024, 1024)
    assert rows == 4096
    one = vsb.encoder_activation_bytes(cfg, rows)
    assert one == rows * (3 * 1152 + 1152 + 1152 + 4304) * 2
    assert one < 100 * (1 << 20)  # ~78 MiB, not ~2 GiB
    assert one * cfg.depth > 2 * (1 << 30)


def test_the_embedding_post_is_the_designs_exact_ten_MiB():
    cfg = vs.VisionEncoderConfig()
    rows = cfg.patch_rows(1024, 1024)
    assert vsb.embedding_bytes_for(cfg, rows) == (4096 // 4) * 5120 * 2
    assert vsb.embedding_bytes_for(cfg, rows) == 10 * (1 << 20)


def test_the_booked_geometry_is_the_acceptance_image_and_is_carried(model_dir):
    assert vsb.BOOKED_IMAGE_HW == (1024, 1024)
    service, _ctx, cfg = _build(model_dir)
    rows = cfg.patch_rows(*vsb.BOOKED_IMAGE_HW)
    assert service.tower.activation_bytes == vsb.encoder_activation_bytes(cfg, rows)
    assert service.tower.embedding_bytes == vsb.embedding_bytes_for(cfg, rows)


def test_nothing_is_rounded_up_into_a_reserve(model_dir):
    """"Reserven NIE, nicht ein Byte": every post is an arithmetic result."""
    service, ctx, cfg = _build(model_dir)
    rows = cfg.patch_rows(*vsb.BOOKED_IMAGE_HW)
    assert service.tower.total_bytes == (
        TOWER_BYTES
        + ctx.ctx_bytes
        + vsb.encoder_activation_bytes(cfg, rows)
        + vsb.embedding_bytes_for(cfg, rows)
    )


def test_a_scattered_tower_refuses_rather_than_implying_one_read(tmp_path):
    """A gap inside the extent breaks the single-sequential-read cost model."""
    header = {
        "model.visual.a": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
        "model.text.pad": {"dtype": "BF16", "shape": [64], "data_offsets": [8, 4104]},
        "model.visual.b": {"dtype": "BF16", "shape": [4], "data_offsets": [4104, 4112]},
    }
    blob = json.dumps(header).encode()
    shard = tmp_path / "model-00001-of-00001.safetensors"
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * 4112)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in header}})
    )
    with pytest.raises(vs.VisionStageTowerUnreadable):
        _build(str(tmp_path))


def test_a_checkpoint_with_no_tower_refuses_before_any_child_is_spawned(tmp_path):
    """Cheap-first ordering: headers are read before CUDA is ever touched."""
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "s.safetensors"}})
    )
    spawned = []

    def never(argv, env):
        spawned.append(argv)
        return _probe_ok(argv, env)

    with pytest.raises(vsl.VisionStageLoadRefused):
        _build(str(tmp_path), context_runner=never)
    assert spawned == []


# --------------------------------- 3. no service under transient = REFUSAL --


def _item(feature=object(), embeddings=None):
    return types.SimpleNamespace(feature=feature, precomputed_embeddings=embeddings)


def test_arming_hands_the_images_to_the_RANK_stage(model_dir, monkeypatch):
    """User design 2026-09-24: the stage runs inside the P group's PP0 rank
    (weg2/vision_rank_runner.py) and REPLACES the tokenizer-process stage.
    Arming builds no service here, and the seam passes the pixels on."""
    monkeypatch.setenv(vsb.VISION_ENV, "transient")
    monkeypatch.setenv(vsb.VISION_GROUP_ENV, "P")
    armed = vsb.arm_transient_vision(
        types.SimpleNamespace(model_path=model_dir),
        types.SimpleNamespace(hf_config=_hf_config()),
    )
    assert armed is True
    assert vss.installed() is None
    assert vss.rank_stage_installed()
    assert vss.arm_refusal() == ""
    item = _item()
    assert vss.maybe_run([item]) is None
    vss.assert_nothing_unstaged([item])  # the rank stages it: no refusal here
    assert item.feature is not None and item.precomputed_embeddings is None


def test_the_rank_stage_state_is_cleared_by_a_refusal_and_by_a_service(model_dir):
    vss.install_rank_stage()
    vss.install_refusal("a later refusal")
    assert not vss.rank_stage_installed()
    vss.install_rank_stage()
    service, _ctx, _cfg = _build(model_dir)
    vss.install(service)
    assert not vss.rank_stage_installed()


def test_arming_without_a_vision_config_refuses_by_name(model_dir, monkeypatch):
    monkeypatch.setenv(vsb.VISION_ENV, "transient")
    monkeypatch.setenv(vsb.VISION_GROUP_ENV, "P")
    assert vsb.arm_transient_vision(
        types.SimpleNamespace(model_path=model_dir),
        types.SimpleNamespace(hf_config=types.SimpleNamespace(rms_norm_eps=1e-6)),
    ) is False
    assert "vision_config" in vss.arm_refusal()
    with pytest.raises(vss.VisionStageNotArmed):
        vss.maybe_run([_item()])


def test_an_image_with_NO_service_under_transient_refuses_BY_NAME(caplog):
    """The whole point of this slice: a silent no-op is not an option."""
    with caplog.at_level(logging.ERROR):
        vss.install_refusal("VisionStageArmRefused: no model path")
    assert any(vss.W_ARM_REFUSED in r.getMessage() for r in caplog.records)

    with pytest.raises(vss.VisionStageNotArmed) as e:
        vss.maybe_run([_item()])
    msg = str(e.value)
    assert vss.W_NOT_ARMED in msg
    assert "no model path" in msg  # the BOOT reason, quoted at request time


def test_a_failed_arming_records_the_reason_and_does_NOT_raise(monkeypatch, caplog):
    monkeypatch.setenv(vsb.VISION_ENV, "transient")
    monkeypatch.setenv(vsb.VISION_GROUP_ENV, "P")
    with caplog.at_level(logging.ERROR):
        out = vsb.arm_transient_vision(
            types.SimpleNamespace(model_path=""),
            types.SimpleNamespace(hf_config=_hf_config()),
        )
    assert out is False
    assert vss.installed() is None
    assert not vss.rank_stage_installed()
    assert "no model path" in vss.arm_refusal()
    assert any(vss.W_ARM_REFUSED in r.getMessage() for r in caplog.records)


def test_a_transient_boot_without_an_mm_processor_still_serves_TEXT(monkeypatch):
    """An arming failure makes the IMAGE path loud; it does not kill the boot."""
    monkeypatch.setenv(vsb.VISION_ENV, "transient")
    monkeypatch.setenv(vsb.VISION_GROUP_ENV, "P")
    assert vsb.arm_transient_vision(
        types.SimpleNamespace(model_path="/nope"),
        types.SimpleNamespace(hf_config=_hf_config()),
        multimodal=False,
    ) is False
    assert "NO multimodal processor" in vss.arm_refusal()
    assert vss.maybe_run([]) is None                      # text: no items
    assert vss.maybe_run([_item(feature=None)]) is None   # text: nothing stageable
    with pytest.raises(vss.VisionStageNotArmed):
        vss.maybe_run([_item()])                          # image: named refusal


def test_a_boot_that_is_not_transient_arms_nothing(model_dir, monkeypatch):
    monkeypatch.delenv(vsb.VISION_ENV, raising=False)
    assert vsb.arm_transient_vision(
        types.SimpleNamespace(model_path=model_dir),
        types.SimpleNamespace(hf_config=_hf_config()),
    ) is False
    assert not vss.rank_stage_installed() and vss.arm_refusal() == ""


def test_an_item_that_already_carries_rows_is_not_a_refusal():
    """A re-entry, or an encoder-disagg boot: nothing to stage, nothing to say."""
    vss.install_refusal("whatever")
    assert vss.maybe_run([_item(embeddings=object())]) is None


def test_with_no_transient_boot_at_all_an_image_is_NOT_refused_here():
    """`off`/`resident` route elsewhere; this module must stay out of the way."""
    assert vss.arm_refusal() == ""
    assert vss.maybe_run([_item()]) is None


def test_installing_a_service_clears_a_recorded_refusal(model_dir):
    vss.install_refusal("an earlier attempt")
    service, _ctx, _cfg = _build(model_dir)
    vss.install(service)
    assert vss.arm_refusal() == ""
    assert vss.maybe_run([_item(feature=None)]) is None


def test_the_processor_seam_does_NOT_swallow_the_not_armed_refusal():
    """It swallows everything else on purpose; swallowing this one would be
    the silent shape the arming path exists to end."""
    import inspect

    from sglang.srt.multimodal.processors import base_processor as bp

    src = inspect.getsource(bp.BaseMultimodalProcessor.process_and_combine_mm_data)
    assert "_vss.maybe_run(all_collected_items)" in src
    # The re-raise clause now names TWO classes -- `VisionStageRequestRefused`
    # joined it when the xsn405 fix made a refused stage terminal at the seam
    # (see test_vision_stage_census_refusal_0920) -- so this pins the CLAUSE,
    # not one spelling of it, and still pins that it comes BEFORE the
    # swallow-everything-else clause.
    assert "_vss.VisionStageNotArmed" in src
    reraise = src.index("_vss.VisionStageNotArmed, _vss.VisionStageRequestRefused")
    assert reraise < src.index("vision stage seam skipped")


def test_the_not_armed_class_is_not_a_refusal_subclass():
    """A caller mapping `VisionStageRefused` to 501 must not catch this."""
    assert not issubclass(vss.VisionStageNotArmed, vs.VisionStageRefused)


# ---------------------------------------------- 4. W103 video, unchanged --


@pytest.mark.parametrize("mode", ["off", "resident", "transient"])
def test_video_still_refuses_in_every_mode(mode):
    verdict, why = fr.vision_verdict(image_parts=0, video_parts=1, mode=mode)
    assert verdict == fr.VERDICT_REFUSE_VIDEO
    assert "video" in why.lower()


def test_video_beats_image_under_transient_too():
    verdict, _ = fr.vision_verdict(image_parts=1, video_parts=1, mode="transient")
    assert verdict == fr.VERDICT_REFUSE_VIDEO


def test_the_video_w_code_is_still_W103():
    import inspect

    src = inspect.getsource(fr.Front.handle_generate)
    assert '"W103 Weg2VideoRefused"' in src


# -------------------------------- 5. without the flag, nothing is different --


def test_the_arming_variable_is_published_for_P_TRANSIENT_ONLY():
    def env(group, vision):
        return lz.build_env(
            "/t", "/v", "0,1,2", "/s", False, "tag", group=group, vision=vision
        ).get(lz.VISION_STAGE_ENV)

    assert env("P", "transient") == "transient"
    assert env("D", "transient") is None   # D has no use for a tower or a probe
    assert env("P", "off") is None
    assert env("P", "resident") is None


def test_an_inherited_shell_value_is_POPPED_not_left_to_arm_a_stage(monkeypatch):
    """R19 discipline: this is launcher OUTPUT, never operator input."""
    monkeypatch.setenv(lz.VISION_STAGE_ENV, "transient")
    env = lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group="P", vision="off")
    assert lz.VISION_STAGE_ENV not in env


def test_the_publisher_and_the_reader_use_ONE_key():
    assert lz.VISION_STAGE_ENV == vsb.VISION_ENV == "SGLANG_WEG2_VISION"


def test_the_launcher_default_publishes_nothing():
    """`build_env` called the way every pre-#58 caller called it."""
    env = lz.build_env("/t", "/v", "0,1,2", "/s", False, "tag", group="P")
    assert lz.VISION_STAGE_ENV not in env


@pytest.mark.parametrize("env", [
    {},
    {"SGLANG_WEG2_VISION": ""},
    {"SGLANG_WEG2_VISION": "off"},
    {"SGLANG_WEG2_VISION": "resident"},
    {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "D"},
])
def test_arming_is_a_no_op_outside_P_transient(env):
    assert vsb.vision_mode(env) == ""


def test_the_mode_is_transient_for_P_and_for_a_group_less_process():
    assert vsb.vision_mode({"SGLANG_WEG2_VISION": "transient"}) == "transient"
    assert vsb.vision_mode(
        {"SGLANG_WEG2_VISION": "transient", "SGLANG_WEG2_GROUP": "P"}
    ) == "transient"


def test_without_the_flag_arming_touches_NO_state_at_all(model_dir):
    """Byte-for-byte the old path: no install, no refusal, no rank handover."""
    out = vsb.arm_transient_vision(
        types.SimpleNamespace(model_path=model_dir),
        types.SimpleNamespace(hf_config=_hf_config()),
        env={},
    )
    assert out is False
    assert vss.installed() is None
    assert not vss.rank_stage_installed()
    assert vss.arm_refusal() == ""


def test_the_model_argv_per_vision_form_is_unchanged_by_this_slice():
    assert lz.vision_model_flags("off") == ["--no-enable-multimodal"]
    assert lz.vision_model_flags("resident") == []
    assert lz.vision_model_flags("transient") == [
        "--json-model-override-args", '{"language_model_only": true}'
    ]


@pytest.mark.parametrize("mode", ["off", "resident", "transient"])
def test_a_text_only_request_routes_in_every_mode(mode):
    verdict, _ = fr.vision_verdict(image_parts=0, video_parts=0, mode=mode)
    assert verdict == fr.VERDICT_ROUTE


def test_the_tokenizer_arms_from_its_own_process_after_BOTH_branches():
    """OUTSIDE the `is_multimodal` branch, and that placement is the point.

    Inside it, a transient boot whose model config came back NOT multimodal
    would arm nothing and say nothing -- the silent shape. Out here the case
    reaches `arm_transient_vision`, which refuses it by name.
    """
    import inspect

    from sglang.srt.managers import tokenizer_manager as tm

    src = inspect.getsource(tm.TokenizerManager.init_tokenizer_and_processor)
    assert "arm_transient_vision(" in src
    head, _, _tail = src.partition("arm_transient_vision(")
    assert "self.mm_processor = get_mm_processor(" in head     # the if branch
    assert "self.mm_processor = self.processor = None" in head  # the else branch
    assert "multimodal=bool(self.mm_processor is not None)" in src


def test_a_transient_boot_with_no_multimodal_processor_REFUSES_by_name(model_dir):
    """`--no-enable-multimodal` + transient: no mm_items ever, so no stage ever.

    This is the config mistake the two switches exist to keep apart, and it
    would otherwise be invisible until an image came in.
    """
    out = vsb.arm_transient_vision(
        types.SimpleNamespace(model_path=model_dir),
        types.SimpleNamespace(hf_config=_hf_config()),
        multimodal=False,
        env={"SGLANG_WEG2_VISION": "transient"},
    )
    assert out is False
    reason = vss.arm_refusal()
    assert "no multimodal processor" in reason.lower() or "NO multimodal" in reason
    assert "language_model_only" in reason
    assert "--no-enable-multimodal" in reason
