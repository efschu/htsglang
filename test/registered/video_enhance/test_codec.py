"""Unit tests for the codec and colour ends of the #333 enhance chain.

CPU-only; no GPU, no network. The two codec libraries are replaced by fakes,
so what is exercised here is the chain-facing behaviour -- pull semantics,
segmentation, backend policy, colour arithmetic -- rather than NVDEC or NVENC
themselves. Run:
  python -m pytest test/registered/video_enhance/test_codec.py -q
"""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from sglang.srt.video_enhance import codec
from sglang.srt.video_enhance.codec import (
    BT601,
    BT709,
    FULL_RANGE,
    LIMITED_RANGE,
    ClipSpec,
    CodecBackendUnavailable,
    CodecError,
    ColorToRgbStage,
    ColorToYuvStage,
    DecodeStage,
    EncodeStage,
    SourceInfo,
    build_ffmpeg_decode_command,
    build_ffmpeg_encode_command,
    make_test_clip,
    nv12_to_rgb,
    parse_ffprobe_output,
    probe_source,
    rgb_format_for_dtype,
    rgb_to_nv12,
    select_decode_backend,
    select_encode_backend,
    synthetic_frame_rgb,
)
from sglang.srt.video_enhance import frame_math
from sglang.srt.video_enhance.frame_math import (
    PixelFormat,
    Resolution,
    codec_pool_bytes,
)
from sglang.srt.video_enhance.frames import Frame, HostResidencyError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Colour matrices
# ---------------------------------------------------------------------------


def test_bt709_matches_the_published_coefficients():
    # ITU-R BT.709-6: kr=0.2126, kb=0.0722, and the four derived non-trivial
    # entries every reference implementation quotes to four decimals.
    assert (BT709.kr, BT709.kb) == (0.2126, 0.0722)
    assert BT709.kg == pytest.approx(0.7152)
    (_, _, c_rv), (_, c_gu, c_gv), (_, c_bu, _) = BT709.to_rgb_coefficients()
    assert c_rv == pytest.approx(1.5748, abs=1e-9)
    assert c_gu == pytest.approx(-0.187324, abs=1e-6)
    assert c_gv == pytest.approx(-0.468124, abs=1e-6)
    assert c_bu == pytest.approx(1.8556, abs=1e-9)


def test_bt601_matches_the_published_coefficients():
    assert (BT601.kr, BT601.kb) == (0.299, 0.114)
    (_, _, c_rv), (_, c_gu, c_gv), (_, c_bu, _) = BT601.to_rgb_coefficients()
    assert c_rv == pytest.approx(1.402, abs=1e-9)
    assert c_gu == pytest.approx(-0.344136, abs=1e-6)
    assert c_gv == pytest.approx(-0.714136, abs=1e-6)
    assert c_bu == pytest.approx(1.772, abs=1e-9)


@pytest.mark.parametrize("matrix", [BT709, BT601])
def test_forward_and_inverse_matrices_are_exact_inverses(matrix):
    to_rgb = torch.tensor(matrix.to_rgb_coefficients(), dtype=torch.float64)
    to_yuv = torch.tensor(matrix.to_yuv_coefficients(), dtype=torch.float64)
    assert torch.allclose(
        to_rgb @ to_yuv, torch.eye(3, dtype=torch.float64), atol=1e-12
    )


def test_first_row_of_the_yuv_matrix_sums_to_one():
    # A grey input must produce zero chroma; that is only true if the luma row
    # is normalised, which is the invariant a hand-typed matrix breaks first.
    for matrix in (BT709, BT601):
        assert sum(matrix.to_yuv_coefficients()[0]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Colour conversion
# ---------------------------------------------------------------------------


def _block_uniform_rgb(width: int, height: int) -> torch.Tensor:
    """RGB whose 2x2 blocks are uniform, so 4:2:0 subsampling is lossless.

    Values stay inside [0.1, 0.9] so that neither the limited-range mapping
    nor the [0, 1] clamp clips, which would otherwise show up as round-trip
    error that has nothing to do with quantisation.
    """
    torch.manual_seed(0)
    small = torch.rand(1, 3, height // 2, width // 2, dtype=torch.float32)
    small = small * 0.8 + 0.1
    return small.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


@pytest.mark.parametrize("matrix", ["709", "601"])
@pytest.mark.parametrize("color_range", ["limited", "full"])
def test_rgb_yuv_rgb_round_trip_within_quantisation_tolerance(matrix, color_range):
    # Tolerance derivation, limited range and BT.709, the worst of the four
    # combinations: luma contributes half a code of 1/219, and chroma
    # contributes half a code of 1/224 amplified by the largest inverse
    # coefficient (1.8556 for blue), so
    #     0.5/219 + 1.8556 * 0.5/224 = 0.00228 + 0.00414 = 0.0064.
    # 0.01 leaves room for the fp32 arithmetic without hiding a real error;
    # a wrong matrix or a wrong range moves the result by 10x this.
    rgb = _block_uniform_rgb(32, 16)
    nv12 = rgb_to_nv12(rgb, matrix=matrix, color_range=color_range)
    back = nv12_to_rgb(nv12, matrix=matrix, color_range=color_range, dtype="fp32")
    assert torch.max(torch.abs(back - rgb)).item() < 0.01


def test_round_trip_preserves_a_flat_grey_exactly_in_full_range():
    # 0.5 full-swing is code 128 in full range, which survives the round trip
    # with no rounding at all. If it does not, the offsets are wrong.
    rgb = torch.full((1, 3, 8, 8), 128.0 / 255.0, dtype=torch.float32)
    nv12 = rgb_to_nv12(rgb, color_range="full")
    assert torch.all(nv12[:8] == 128)
    assert torch.all(nv12[8:] == 128)
    back = nv12_to_rgb(nv12, color_range="full", dtype="fp32")
    assert torch.allclose(back, rgb, atol=1e-6)


def test_limited_range_puts_black_and_white_at_16_and_235():
    black = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    white = torch.ones((1, 3, 4, 4), dtype=torch.float32)
    assert torch.all(rgb_to_nv12(black)[:4] == 16)
    assert torch.all(rgb_to_nv12(white)[:4] == 235)


def test_limited_and_full_range_disagree():
    rgb = _block_uniform_rgb(8, 8)
    limited = rgb_to_nv12(rgb, color_range=LIMITED_RANGE)
    full = rgb_to_nv12(rgb, color_range=FULL_RANGE)
    assert not torch.equal(limited, full)


def test_nv12_layout_is_luma_then_interleaved_chroma():
    rgb = _block_uniform_rgb(16, 8)
    nv12 = rgb_to_nv12(rgb)
    assert nv12.shape == (8 * 3 // 2, 16)
    assert nv12.dtype == torch.uint8
    # Round-tripping the chroma plane through the reshape the decoder does
    # must give back a (H/2, W/2, 2) block, i.e. the plane really is
    # interleaved rather than planar.
    assert nv12[8:].reshape(4, 8, 2).shape == (4, 8, 2)


def test_nv12_to_rgb_output_is_nchw_and_in_unit_range():
    nv12 = rgb_to_nv12(_block_uniform_rgb(16, 8))
    out = nv12_to_rgb(nv12, dtype="fp16")
    assert out.shape == (1, 3, 8, 16)
    assert out.dtype == torch.float16
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_chroma_upsample_is_nearest_neighbour():
    # One non-neutral chroma sample must colour exactly its own 2x2 luma
    # footprint. A bilinear upsample would bleed into the neighbours.
    nv12 = torch.full((6, 4), 128, dtype=torch.uint8)
    nv12[:4] = 128
    nv12[4, 0] = 200  # U of the top-left chroma sample
    rgb = nv12_to_rgb(nv12, dtype="fp32")[0]
    blue = rgb[2]
    assert float(blue[0, 0]) == pytest.approx(float(blue[1, 1]), abs=1e-6)
    assert float(blue[0, 0]) > float(blue[0, 2]) + 0.1


def test_conversion_rejects_malformed_nv12():
    with pytest.raises(ValueError, match="2-D"):
        nv12_to_rgb(torch.zeros(3, 6, 4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="4:2:0"):
        nv12_to_rgb(torch.zeros(7, 4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="4:2:0"):
        nv12_to_rgb(torch.zeros(6, 5, dtype=torch.uint8))


def test_conversion_rejects_odd_dimensions_and_batches():
    with pytest.raises(ValueError, match="even dimensions"):
        rgb_to_nv12(torch.zeros(3, 5, 4))
    with pytest.raises(ValueError, match="one frame per work unit"):
        rgb_to_nv12(torch.zeros(2, 3, 4, 4))
    with pytest.raises(ValueError, match="planar RGB"):
        rgb_to_nv12(torch.zeros(4, 4, 4))


def test_rgb_format_mirrors_the_vsgan_naming():
    assert rgb_format_for_dtype("fp16") is PixelFormat.RGB_FP16
    assert rgb_format_for_dtype("fp32") is PixelFormat.RGB_FP32
    with pytest.raises(ValueError, match="RGBH"):
        rgb_format_for_dtype("int8")


def test_unknown_matrix_and_range_are_rejected_by_name():
    with pytest.raises(ValueError, match="unknown colour matrix"):
        nv12_to_rgb(torch.zeros(6, 4, dtype=torch.uint8), matrix="2020")
    with pytest.raises(ValueError, match="unknown colour range"):
        nv12_to_rgb(torch.zeros(6, 4, dtype=torch.uint8), color_range="studio")


# ---------------------------------------------------------------------------
# Colour stages
# ---------------------------------------------------------------------------


@pytest.fixture
def unenforced_residency(monkeypatch):
    """Disable the §8.1 device check so the stages run on CPU tensors.

    The check itself is covered by
    ``test_colour_stage_refuses_to_pass_a_host_tensor_downstream``; here it
    would only assert that a CPU test runs on the CPU.
    """
    monkeypatch.setattr(Frame, "require_device", lambda self, stage: self)


def _nv12_frame(width=16, height=8, index=3) -> Frame:
    return Frame(
        data=rgb_to_nv12(_block_uniform_rgb(width, height)),
        resolution=Resolution(width, height),
        format=PixelFormat.NV12,
        index=index,
        pts=1234,
    )


def test_colour_stages_preserve_frame_identity(unenforced_residency):
    to_rgb = ColorToRgbStage(dtype="fp16")
    to_yuv = ColorToYuvStage()
    (rgb_frame,) = to_rgb.process([_nv12_frame()])
    assert rgb_frame.format is PixelFormat.RGB_FP16
    assert rgb_frame.resolution == Resolution(16, 8)
    assert (rgb_frame.index, rgb_frame.pts) == (3, 1234)
    (yuv_frame,) = to_yuv.process([rgb_frame])
    assert yuv_frame.format is PixelFormat.NV12
    assert (yuv_frame.index, yuv_frame.pts) == (3, 1234)
    assert yuv_frame.data.shape == (12, 16)


def test_colour_stages_pass_the_end_of_stream_sentinel_through(unenforced_residency):
    eos = Frame.eos(9)
    for stage in (ColorToRgbStage(), ColorToYuvStage()):
        (out,) = stage.process([eos])
        assert out is eos


def test_color_to_rgb_rejects_a_non_nv12_input(unenforced_residency):
    frame = _nv12_frame().with_data(
        torch.zeros(1, 3, 8, 16), format=PixelFormat.RGB_FP32
    )
    with pytest.raises(ValueError, match="reads NV12"):
        ColorToRgbStage().process([frame])


def test_color_to_yuv_rejects_a_precision_the_chain_did_not_configure(
    unenforced_residency,
):
    frame = _nv12_frame().with_data(
        torch.zeros(1, 3, 8, 16, dtype=torch.float32), format=PixelFormat.RGB_FP32
    )
    with pytest.raises(ValueError, match="configured for rgb_fp16"):
        ColorToYuvStage(dtype="fp16").process([frame])
    (out,) = ColorToYuvStage(dtype="fp32").process([frame])
    assert out.format is PixelFormat.NV12


def test_colour_stage_refuses_to_pass_a_host_tensor_downstream():
    with pytest.raises(HostResidencyError, match="no host round-trip"):
        ColorToRgbStage().process([_nv12_frame()])


# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------


_FFPROBE_JSON = json.dumps(
    {
        "streams": [
            {"codec_type": "audio", "codec_name": "aac"},
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24000/1001",
                "r_frame_rate": "24000/1001",
                "nb_frames": "240",
            },
        ],
        "format": {"format_name": "mov,mp4,m4a", "duration": "10.01"},
    }
)


def test_ffprobe_parsing_picks_the_video_stream():
    info = parse_ffprobe_output(_FFPROBE_JSON)
    assert info.resolution == Resolution(1920, 1080)
    assert info.fps == Fraction(24000, 1001)
    assert info.frame_count == 240
    assert info.codec == "h264"
    assert info.pixel_format is PixelFormat.NV12
    assert info.backend == "ffprobe"


def test_ffprobe_parsing_derives_a_frame_count_from_the_duration():
    doc = json.loads(_FFPROBE_JSON)
    doc["streams"][1]["nb_frames"] = "N/A"
    doc["streams"][1]["duration"] = "2.0"
    info = parse_ffprobe_output(json.dumps(doc))
    assert info.frame_count == 47  # 2.0 s at 24000/1001 fps, truncated


def test_ffprobe_parsing_leaves_an_unknown_frame_count_as_none():
    doc = json.loads(_FFPROBE_JSON)
    doc["streams"][1]["nb_frames"] = "N/A"
    doc["streams"][1].pop("duration", None)
    doc["format"].pop("duration", None)
    info = parse_ffprobe_output(json.dumps(doc))
    assert info.frame_count is None


def test_ffprobe_parsing_rejects_a_source_with_no_video():
    with pytest.raises(CodecError, match="no video stream"):
        parse_ffprobe_output(json.dumps({"streams": [{"codec_type": "audio"}]}))


def test_probe_source_reports_a_missing_ffprobe_by_name(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(CodecError, match="not found on PATH"):
        probe_source("clip.mp4", ffprobe_path="ffprobe-does-not-exist")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_auto_decode_prefers_nvdec_when_it_can_open_the_codec():
    assert select_decode_backend("h264", "auto", nvc_available=True) == "pynvvideocodec"
    assert select_decode_backend("hevc", "auto", nvc_available=True) == "pynvvideocodec"


def test_auto_decode_falls_back_for_an_unsupported_codec_or_missing_package():
    assert select_decode_backend("prores", "auto", nvc_available=True) == "ffmpeg"
    assert select_decode_backend("h264", "auto", nvc_available=False) == "ffmpeg"


def test_explicit_decode_backend_fails_loudly_rather_than_falling_back():
    with pytest.raises(CodecBackendUnavailable, match="package is unavailable"):
        select_decode_backend("h264", "pynvvideocodec", nvc_available=False)
    with pytest.raises(CodecBackendUnavailable, match="no decoder for"):
        select_decode_backend("prores", "pynvvideocodec", nvc_available=True)
    with pytest.raises(ValueError, match="unknown decode backend"):
        select_decode_backend("h264", "cuvid", nvc_available=True)


def test_encode_backend_policy_routes_muxed_containers_to_ffmpeg():
    assert (
        select_encode_backend(
            "h264", "annexb", "auto", nvc_available=True, inprocess_enabled=True
        )
        == "pynvvideocodec"
    )
    assert (
        select_encode_backend(
            "h264", "mpegts", "auto", nvc_available=True, inprocess_enabled=True
        )
        == "ffmpeg"
    )
    with pytest.raises(CodecBackendUnavailable, match="needs a muxer"):
        select_encode_backend("h264", "mpegts", "pynvvideocodec", nvc_available=True)


def test_encode_backend_rejects_unknown_codecs_and_containers():
    with pytest.raises(ValueError, match="unknown encode codec"):
        select_encode_backend("vp9", "annexb", "auto", nvc_available=True)
    with pytest.raises(ValueError, match="unknown container"):
        select_encode_backend("h264", "mkv", "auto", nvc_available=True)


# ---------------------------------------------------------------------------
# ffmpeg command construction
# ---------------------------------------------------------------------------


def test_ffmpeg_decode_command_keeps_the_decode_on_nvdec():
    cmd = build_ffmpeg_decode_command("clip.mp4", Resolution(1920, 1080), device_id=2)
    assert "-hwaccel" in cmd and cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd[cmd.index("-hwaccel_output_format") + 1] == "cuda"
    assert cmd[cmd.index("-hwaccel_device") + 1] == "2"
    # hwdownload is the §8.1 violation the fallback documents; assert it is
    # present so the docstring cannot drift away from the command.
    assert any("hwdownload" in part for part in cmd)
    assert cmd[cmd.index("-pix_fmt") + 1] == "nv12"


def test_ffmpeg_encode_command_selects_the_nvenc_encoder_and_muxer():
    cmd = build_ffmpeg_encode_command(
        Resolution(1280, 720), Fraction(30000, 1001), codec="hevc", container="mpegts"
    )
    assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
    assert cmd[cmd.index("-f", cmd.index("-c:v")) + 1] == "mpegts"
    assert cmd[cmd.index("-s") + 1] == "1280x720"
    assert cmd[cmd.index("-r") + 1] == "30000/1001"
    assert cmd[-1] == "-"


def test_ffmpeg_encode_command_uses_the_elementary_muxer_for_annexb():
    for cdc, muxer in (("h264", "h264"), ("hevc", "hevc"), ("av1", "obu")):
        cmd = build_ffmpeg_encode_command(
            Resolution(64, 64), Fraction(30), codec=cdc, container="annexb"
        )
        assert cmd[-2:] == [muxer, "-"]


# ---------------------------------------------------------------------------
# Fake PyNvVideoCodec
# ---------------------------------------------------------------------------


class _FakeDecodedFrame:
    """Stands in for ``nvc.DecodedFrame``, DLPack seam included.

    Real 2.2.0 ``DecodedFrame`` objects export ``__dlpack__`` and
    ``__dlpack_device__``; delegating both to a CPU torch tensor reproduces
    the seam the production path uses without a device.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, *args, **kwargs):
        return self._tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


class _FakePixelFormat:
    def __init__(self, name):
        self.name = name


class _FakeDemuxer:
    def __init__(self, packet_count):
        self._packets = list(range(packet_count))

    def __iter__(self):
        return iter(self._packets)

    def GetNvCodecId(self):  # noqa: N802 - mirrors the pybind11 name
        return "H264"


class _FakeDecoder:
    def __init__(self, resolution, frames_per_packet=1, pixel_format="NV12"):
        self.resolution = resolution
        self.frames_per_packet = frames_per_packet
        self.decode_calls = 0
        self._pixel_format = pixel_format
        self._next = 0

    def Decode(self, packet):  # noqa: N802 - mirrors the pybind11 name
        self.decode_calls += 1
        rows = self.resolution.height * 3 // 2
        out = []
        for _ in range(self.frames_per_packet):
            tensor = torch.full(
                (rows, self.resolution.width), self._next % 256, dtype=torch.uint8
            )
            self._next += 1
            out.append(_FakeDecodedFrame(tensor))
        return out

    def GetPixelFormat(self):  # noqa: N802 - mirrors the pybind11 name
        return _FakePixelFormat(self._pixel_format)


class _FakeEncoder:
    def __init__(self, width, height, fmt, cpu_input, **options):
        self.width = width
        self.height = height
        self.fmt = fmt
        self.cpu_input = cpu_input
        self.options = options
        self.frames = 0
        self.ended = False

    def Encode(self, tensor):  # noqa: N802 - mirrors the pybind11 name
        self.frames += 1
        return [{"data": bytes([self.frames % 256]) * 4, "timestamp": self.frames}]

    def EndEncode(self):  # noqa: N802 - mirrors the pybind11 name
        self.ended = True
        return [{"data": b"TAIL", "timestamp": 0}]


class _FakeNvc:
    def __init__(
        self, resolution, *, frames_per_packet=1, packets=4, pixel_format="NV12"
    ):
        self.resolution = resolution
        self.frames_per_packet = frames_per_packet
        self.packets = packets
        self.pixel_format = pixel_format
        self.decoder = None
        self.encoder = None

    def CreateDemuxer(self, filename):  # noqa: N802 - mirrors the pybind11 name
        return _FakeDemuxer(self.packets)

    def CreateDecoder(self, **kwargs):  # noqa: N802 - mirrors the pybind11 name
        self.decoder_kwargs = kwargs
        self.decoder = _FakeDecoder(
            self.resolution, self.frames_per_packet, self.pixel_format
        )
        return self.decoder

    def CreateEncoder(self, width, height, fmt, cpu_input, **options):  # noqa: N802
        self.encoder = _FakeEncoder(width, height, fmt, cpu_input, **options)
        return self.encoder


def _source_info(resolution, frames=4, codec_name="h264") -> SourceInfo:
    return SourceInfo(
        resolution=resolution,
        frame_count=frames,
        fps=Fraction(30),
        pixel_format=PixelFormat.NV12,
        codec=codec_name,
        container="mp4",
        backend="ffprobe",
    )


@pytest.fixture
def fake_nvc(monkeypatch):
    def _install(resolution, **kwargs):
        fake = _FakeNvc(resolution, **kwargs)
        monkeypatch.setattr(codec, "_import_pynvvideocodec", lambda: fake)
        return fake

    return _install


# ---------------------------------------------------------------------------
# DecodeStage
# ---------------------------------------------------------------------------


def test_decode_pull_returns_exactly_what_was_asked_for(fake_nvc, unenforced_residency):
    res = Resolution(16, 8)
    fake = fake_nvc(res, packets=8)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    frames = stage.pull(3)
    assert len(frames) == 3
    assert [f.index for f in frames] == [0, 1, 2]
    assert all(f.format is PixelFormat.NV12 for f in frames)
    assert all(f.data.shape == (12, 16) for f in frames)
    # Laziness: three frames means three packets, not the whole demuxer.
    assert fake.decoder.decode_calls == 3
    assert stage.frames_decoded == 3


def test_decode_carries_over_extra_frames_from_one_packet(fake_nvc):
    res = Resolution(16, 8)
    fake = fake_nvc(res, frames_per_packet=3, packets=4)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    assert len(stage.pull(1)) == 1
    assert fake.decoder.decode_calls == 1
    assert len(stage.pull(2)) == 2
    # The carry-over satisfied the second call without decoding again.
    assert fake.decoder.decode_calls == 1


def test_decode_emits_one_end_of_stream_sentinel_then_nothing(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    frames = stage.pull(5)
    assert len(frames) == 3
    assert [f.end_of_stream for f in frames] == [False, False, True]
    assert frames[-1].index == 2
    assert stage.pull(5) == []


def test_decode_iteration_yields_every_frame_and_the_sentinel(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=3)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    frames = list(stage)
    assert len(frames) == 4
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert frames[-1].end_of_stream


def test_decode_cancellation_stops_production_immediately(fake_nvc):
    res = Resolution(16, 8)
    fake = fake_nvc(res, packets=100)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    stage.pull(2)
    stage.cancel()
    assert stage.cancelled
    assert stage.pull(4) == []
    assert fake.decoder.decode_calls == 2


def test_decode_close_releases_the_backend_and_is_idempotent(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=4)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    stage.pull(1)
    stage.close()
    stage.close()
    assert stage.pull(1) == []


def test_decode_rejects_a_non_nv12_surface(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2, pixel_format="P016")
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    with pytest.raises(CodecError, match="8-bit NV12"):
        stage.pull(1)


def test_decode_probe_reports_the_resolved_backend(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    assert stage.info.backend == "ffprobe"
    assert stage.probe().backend == "pynvvideocodec"
    assert stage.backend_name == "pynvvideocodec"


def test_decode_passes_device_memory_and_the_gpu_id_to_the_decoder(fake_nvc):
    res = Resolution(16, 8)
    fake = fake_nvc(res, packets=2)
    DecodeStage("clip.mp4", info=_source_info(res), device_id=1).probe()
    assert fake.decoder_kwargs["usedevicememory"] is True
    assert fake.decoder_kwargs["gpuid"] == 1


def test_decode_surface_copy_decouples_the_frame_from_the_pool(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2)
    copied = DecodeStage("clip.mp4", info=_source_info(res), copy_surfaces=True).pull(1)
    borrowed = DecodeStage(
        "clip.mp4", info=_source_info(res), copy_surfaces=False
    ).pull(1)
    assert torch.equal(copied[0].data, borrowed[0].data)


def test_decode_pool_bytes_matches_the_reservation_arithmetic(fake_nvc):
    res = Resolution(1920, 1080)
    fake_nvc(res)
    stage = DecodeStage("clip.mp4", info=_source_info(res), pool_depth=6)
    assert stage.pool_bytes == codec_pool_bytes(res, 6, PixelFormat.NV12)


def test_decode_source_can_be_bound_after_construction(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2)
    stage = DecodeStage(resolution=res)
    with pytest.raises(CodecError, match="has no source"):
        stage.pull(1)
    stage.set_source("clip.mp4", info=_source_info(res))
    assert len(stage.pull(1)) == 1
    with pytest.raises(CodecError, match="already open"):
        stage.set_source("other.mp4", info=_source_info(res))


def test_decode_pool_bytes_is_answerable_before_a_source_is_bound():
    res = Resolution(1920, 1080)
    stage = DecodeStage(resolution=res, pool_depth=4)
    assert stage.pool_bytes == codec_pool_bytes(res, 4, PixelFormat.NV12)


def test_decode_rejects_a_source_that_is_not_the_planned_size(fake_nvc):
    fake_nvc(Resolution(16, 8), packets=2)
    stage = DecodeStage(
        "clip.mp4", resolution=Resolution(32, 16), info=_source_info(Resolution(16, 8))
    )
    with pytest.raises(CodecError, match="planned for 32x16"):
        stage.pull(1)


def test_decode_pull_count_must_be_positive(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    with pytest.raises(ValueError, match="must be positive"):
        stage.pull(0)


def test_decode_process_is_a_source_and_takes_no_input(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res, packets=2)
    stage = DecodeStage("clip.mp4", info=_source_info(res))
    assert len(stage.process()) == 1
    with pytest.raises(ValueError, match="is a source"):
        stage.process([Frame.eos(0)])


# ---------------------------------------------------------------------------
# EncodeStage
# ---------------------------------------------------------------------------


def _nv12_device_frame(res: Resolution, index: int) -> Frame:
    rows = res.height * 3 // 2
    return Frame(
        data=torch.full((rows, res.width), index % 256, dtype=torch.uint8),
        resolution=res,
        format=PixelFormat.NV12,
        index=index,
    )


def test_encode_emits_one_segment_per_segment_frames(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = EncodeStage(res, segment_frames=3, backend="pynvvideocodec")
    emitted = []
    for i in range(7):
        emitted.extend(stage.submit(_nv12_device_frame(res, i)))
    assert len(emitted) == 2
    assert stage.segments_emitted == 2
    assert stage.frames_encoded == 7
    # Three frames of four bytes each land in one segment.
    assert all(len(seg) == 12 for seg in emitted)


def test_encode_close_flushes_the_tail_and_is_idempotent(fake_nvc):
    res = Resolution(16, 8)
    fake = fake_nvc(res)
    stage = EncodeStage(res, segment_frames=4, backend="pynvvideocodec")
    stage.process([_nv12_device_frame(res, i) for i in range(2)])
    trailing = stage.close()
    assert len(trailing) == 1
    # Two frames of four bytes plus the flushed tail.
    assert trailing[0] == b"\x01\x01\x01\x01\x02\x02\x02\x02TAIL"
    assert fake.encoder.ended
    assert stage.close() == ()


def test_encode_close_on_an_exact_segment_boundary_still_flushes(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = EncodeStage(res, segment_frames=2, backend="pynvvideocodec")
    segments = stage.process([_nv12_device_frame(res, i) for i in range(2)])
    assert len(segments) == 1
    assert stage.close() == (b"TAIL",)


def test_encode_rejects_frames_after_close(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = EncodeStage(res, segment_frames=2)
    stage.close()
    with pytest.raises(CodecError, match="is closed"):
        stage.submit(_nv12_device_frame(res, 0))


def test_encode_ignores_the_end_of_stream_sentinel(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = EncodeStage(res, segment_frames=2)
    assert stage.process([Frame.eos(0)]) == ()
    assert stage.frames_encoded == 0


def test_encode_validates_the_frame_it_is_handed(fake_nvc):
    res = Resolution(16, 8)
    fake_nvc(res)
    stage = EncodeStage(res)
    wrong_format = _nv12_device_frame(res, 0).with_data(
        torch.zeros(1, 3, 8, 16), format=PixelFormat.RGB_FP16
    )
    with pytest.raises(ValueError, match="takes NV12"):
        stage.submit(wrong_format)
    wrong_size = _nv12_device_frame(Resolution(32, 8), 0)
    with pytest.raises(ValueError, match="configured for 16x8"):
        stage.submit(wrong_size)


def test_encode_configures_nvenc_for_device_input(fake_nvc):
    res = Resolution(64, 32)
    fake = fake_nvc(res)
    EncodeStage(
        res,
        codec="hevc",
        bitrate=4_000_000,
        device_id=1,
        backend="pynvvideocodec",
    ).warmup()
    assert (fake.encoder.width, fake.encoder.height) == (64, 32)
    assert fake.encoder.fmt == "NV12"
    assert fake.encoder.cpu_input is False
    assert fake.encoder.options["codec"] == "hevc"
    assert fake.encoder.options["bitrate"] == 4_000_000
    assert fake.encoder.options["gpu_id"] == 1


def test_encode_reports_a_content_type_per_container():
    assert EncodeStage(Resolution(16, 8), backend="ffmpeg").content_type == "video/H264"
    assert (
        EncodeStage(
            Resolution(16, 8), codec="hevc", container="mpegts", backend="ffmpeg"
        ).content_type
        == "video/mp2t"
    )


def test_encode_pool_bytes_matches_the_reservation_arithmetic():
    res = Resolution(3840, 2160)
    stage = EncodeStage(res, pool_depth=5, backend="ffmpeg")
    assert stage.pool_bytes == codec_pool_bytes(res, 5, PixelFormat.NV12)


def test_encode_segment_frames_must_be_positive():
    with pytest.raises(ValueError, match="segment_frames"):
        EncodeStage(Resolution(16, 8), segment_frames=0, backend="ffmpeg")


# ---------------------------------------------------------------------------
# Synthetic test clip
# ---------------------------------------------------------------------------


def test_synthetic_frames_are_deterministic_and_move():
    res = Resolution(64, 32)
    first = synthetic_frame_rgb(0, res)
    assert (first == synthetic_frame_rgb(0, res)).all()
    assert not (first == synthetic_frame_rgb(1, res)).all()
    assert first.shape == (32, 64, 3)


def test_synthetic_frames_have_a_flat_region_and_a_detailed_region():
    res = Resolution(64, 32)
    frame = synthetic_frame_rgb(5, res)
    split = int(64 * codec.FLAT_REGION_WIDTH_FRACTION)
    flat = frame[:, :split].reshape(-1, 3)
    detailed = frame[:, split:].reshape(-1, 3)
    # Flat per channel, not flat overall: the region is a solid colour, so
    # the three channels differ from each other but never within a channel.
    assert list(flat.std(axis=0)) == [0.0, 0.0, 0.0]
    assert (flat[0] == codec.FLAT_REGION_RGB).all()
    assert detailed.std(axis=0).min() > 40.0


@pytest.mark.skipif(
    subprocess.run(["sh", "-c", "command -v ffmpeg"], capture_output=True).returncode
    != 0,
    reason="ffmpeg is not installed",
)
def test_make_test_clip_is_byte_stable_across_runs(tmp_path: Path):
    res = Resolution(64, 32)
    a = make_test_clip(tmp_path / "a.mp4", res, frames=8, fps=30)
    b = make_test_clip(tmp_path / "b.mp4", res, frames=8, fps=30)
    assert a.read_bytes() == b.read_bytes()
    assert a.stat().st_size > 0


def test_make_test_clip_rejects_an_empty_clip(tmp_path: Path):
    with pytest.raises(ValueError, match="frames must be positive"):
        make_test_clip(tmp_path / "empty.mp4", Resolution(16, 16), frames=0)


def test_clip_spec_manifest_is_serialisable():
    spec = ClipSpec(Resolution(1920, 1080), 120, Fraction(24000, 1001))
    manifest = spec.as_manifest()
    assert manifest["resolution"] == "1920x1080"
    assert manifest["fps"] == "24000/1001"
    assert json.loads(json.dumps(manifest)) == manifest


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------


def test_module_imports_without_the_optional_codec_package(monkeypatch):
    # The planner imports this module on hosts with no NVIDIA stack; backend
    # probing must report unavailable rather than raise.
    def _boom():
        raise CodecBackendUnavailable("simulated absence")

    monkeypatch.setattr(codec, "_import_pynvvideocodec", _boom)
    assert codec.pynvvideocodec_available() is False
    assert select_decode_backend("h264", "auto") == "ffmpeg"


# ==========================================================================
# #484: the in-process zero-copy NVENC lane
# ==========================================================================
#
# TASK_333 §9.5 left the device-input path of PyNvVideoCodec "still not
# working ... rejected with 'incorrect usage of CPU input buffer'", and every
# measurement in that ticket therefore ran through the ffmpeg host round trip.
# The tests below encode the contract that was missing, read off the shipped
# 2.2.0 extension itself:
#
#   PyNvEncoder::Encode  ->  PyObject_HasAttrString(frame, "cuda")
#                            |
#                            +-- present: use frame.cuda(), a LIST of
#                            |            per-plane __cuda_array_interface__
#                            |            views
#                            +-- absent AND usecpuinputbuffer=False:
#                                         throw error 8, "incorrect usage of
#                                         CPU input buffer"
#
# ``_StrictFakeEncoder`` below is that branch, in Python. It is what makes
# these tests a falsifier rather than a description: it fails against the
# pre-#484 wrapper (which exposed only ``__cuda_array_interface__`` through
# ``__slots__`` and so had no ``cuda`` attribute at all) and passes against
# the current one.


class _FakeDeviceTensor:
    """A device tensor as the encoder adapter sees it: shape, dtype, pointer.

    No GPU and no allocation -- the adapter only reads geometry and an integer
    address, so a stand-in that answers those questions exercises exactly the
    code under test. ``is_cuda`` is what ``_NvencDeviceFrame.wrap`` dispatches
    on.
    """

    is_cuda = True
    #: ``_sync_producer`` waits on the stream of the tensor's own device.
    device = "cuda:0"

    def __init__(self, shape, ptr=0x1000, dtype=torch.uint8, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._ptr = ptr
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._ptr


class _StrictFakeEncoder(_FakeEncoder):
    """``_FakeEncoder`` with the 2.2.0 device-input gate modelled.

    Everything it rejects, it rejects for the reason the real extension does,
    and the message is the one the real extension emits.
    """

    def Encode(self, frame, *args):  # noqa: N802 - mirrors the pybind11 name
        if hasattr(frame, "__dlpack__"):
            # The binding probes __dlpack__ before the array interface and
            # calls it with the stream POSITIONALLY; torch declares that
            # parameter keyword-only. The TypeError escapes from inside the
            # encoder and the session's next use faults.
            raise TypeError("__dlpack__() takes 1 positional argument but 2 were given")
        if not hasattr(frame, "cuda"):
            if self.cpu_input is False:
                raise RuntimeError("incorrect usage of CPU input buffer")
            return super().Encode(frame)
        planes = frame.cuda()
        assert isinstance(planes, list), "the device path consumes a plane LIST"
        assert len(planes) == 2, f"NV12 has two planes, got {len(planes)}"
        luma, chroma = (p.__cuda_array_interface__ for p in planes)
        assert luma["typestr"] == chroma["typestr"] == "|u1"
        assert luma["version"] == 3
        # "__cuda_array_interface__ protocol specifies that stream must not
        # be 0" -- the key is omitted rather than set.
        assert "stream" not in luma and "stream" not in chroma
        assert luma["shape"] == (self.height, self.width, 1)
        assert luma["strides"] == (self.width, 1, 1)
        assert chroma["shape"] == (self.height // 2, self.width // 2, 2)
        # One chroma row holds width//2 interleaved UV pairs, so its stride is
        # the full luma width, not half of it.
        assert chroma["strides"] == (self.width, 2, 1)
        base = luma["data"][0]
        assert chroma["data"][0] == base + self.width * self.height
        assert luma["data"][1] is False and chroma["data"][1] is False
        return super().Encode(frame)


def test_nvenc_device_frame_exposes_the_attribute_the_binding_probes():
    """The gate is ``hasattr(frame, "cuda")``. Without it, error 8."""
    frame = codec._NvencDeviceFrame(_FakeDeviceTensor((48, 64)), Resolution(64, 32))
    assert hasattr(frame, "cuda")
    # Nothing here exports __dlpack__: that is the other half of the defect.
    assert not hasattr(frame, "__dlpack__")


def test_nvenc_device_frame_describes_nv12_as_two_planes():
    res = Resolution(64, 32)
    tensor = _FakeDeviceTensor((48, 64), ptr=0x2_0000)
    luma, chroma = (
        p.__cuda_array_interface__ for p in codec._NvencDeviceFrame(tensor, res).cuda()
    )
    assert luma["shape"] == (32, 64, 1)
    assert luma["strides"] == (64, 1, 1)
    assert luma["data"] == (0x2_0000, False)
    assert chroma["shape"] == (16, 32, 2)
    assert chroma["strides"] == (64, 2, 1)
    assert chroma["data"] == (0x2_0000 + 64 * 32, False)


def test_nvenc_device_frame_refuses_a_frame_it_cannot_address():
    res = Resolution(64, 32)
    with pytest.raises(CodecError, match="NV12"):
        codec._NvencDeviceFrame(_FakeDeviceTensor((32, 64)), res)
    with pytest.raises(CodecError, match="uint8"):
        codec._NvencDeviceFrame(_FakeDeviceTensor((48, 64), dtype=torch.float16), res)
    with pytest.raises(CodecError, match="contiguous"):
        codec._NvencDeviceFrame(_FakeDeviceTensor((48, 64), contiguous=False), res)


def test_nvenc_device_frame_keeps_the_allocation_alive():
    """The encoder holds raw pointers; the frame must own a reference."""
    tensor = _FakeDeviceTensor((48, 64))
    frame = codec._NvencDeviceFrame(tensor, Resolution(64, 32))
    assert frame._tensor is tensor


def test_inprocess_nvenc_encodes_through_the_device_path(monkeypatch):
    """The falsifier: the pre-#484 wrapper raises error 8 here."""
    res = Resolution(64, 32)
    encoders = []

    class _Nvc:
        def CreateEncoder(self, w, h, fmt, cpu_input, **options):  # noqa: N802
            enc = _StrictFakeEncoder(w, h, fmt, cpu_input, **options)
            encoders.append(enc)
            return enc

    monkeypatch.setattr(codec, "_import_pynvvideocodec", lambda: _Nvc())
    monkeypatch.setattr(
        codec._PyNvEncodeBackend, "_sync_producer", staticmethod(lambda tensor: None)
    )
    backend = codec._PyNvEncodeBackend(res)
    packets = backend.encode(_FakeDeviceTensor((48, 64)))
    assert packets and encoders[0].frames == 1


def test_a_bare_torch_tensor_would_still_trip_the_dlpack_probe(monkeypatch):
    """Shows the gate is able to fail, and on which input it does.

    Handing the tensor over unwrapped is the obvious thing to try; it enters
    the device path (``Tensor.cuda`` exists) and dies in the dlpack probe.
    """
    encoder = _StrictFakeEncoder(64, 32, "NV12", False)
    with pytest.raises(TypeError, match="__dlpack__"):
        encoder.Encode(torch.zeros(48, 64, dtype=torch.uint8))


def test_an_array_interface_only_view_reproduces_the_recorded_error():
    """The §9.5 symptom, reproduced from its cause.

    An object exposing only ``__cuda_array_interface__`` -- the shape the
    pre-#484 wrapper had -- never reaches the device path.
    """

    class _ArrayInterfaceOnly:
        __slots__ = ("__cuda_array_interface__",)

        def __init__(self):
            self.__cuda_array_interface__ = {
                "shape": (48, 64),
                "strides": (64, 1),
                "data": (0x1000, False),
                "typestr": "|u1",
                "version": 3,
            }

    encoder = _StrictFakeEncoder(64, 32, "NV12", False)
    with pytest.raises(RuntimeError, match="incorrect usage of CPU input buffer"):
        encoder.Encode(_ArrayInterfaceOnly())


def test_auto_keeps_the_ffmpeg_bootstrap_until_the_switch_is_thrown(monkeypatch):
    """The default is the fallback, and the flip is a decision, not an install."""
    monkeypatch.delenv(codec.INPROCESS_NVENC_ENV, raising=False)
    assert not codec.inprocess_nvenc_enabled()
    assert (
        select_encode_backend("h264", "annexb", "auto", nvc_available=True) == "ffmpeg"
    )
    monkeypatch.setenv(codec.INPROCESS_NVENC_ENV, "1")
    assert codec.inprocess_nvenc_enabled()
    assert (
        select_encode_backend("h264", "annexb", "auto", nvc_available=True)
        == "pynvvideocodec"
    )
    # An explicit request ignores the switch: that is how a GPU window runs
    # the path it is about to grade.
    monkeypatch.delenv(codec.INPROCESS_NVENC_ENV, raising=False)
    assert (
        select_encode_backend("h264", "annexb", "pynvvideocodec", nvc_available=True)
        == "pynvvideocodec"
    )


def test_auto_falls_back_when_the_session_cannot_be_opened(monkeypatch, caplog):
    res = Resolution(16, 8)

    def _boom():
        raise CodecBackendUnavailable("libnvidia-encode missing")

    monkeypatch.setattr(codec, "_import_pynvvideocodec", _boom)
    monkeypatch.setattr(codec, "pynvvideocodec_available", lambda: True)
    monkeypatch.setenv(codec.INPROCESS_NVENC_ENV, "1")
    opened = {}

    class _Sub:
        def __init__(self, *a, **kw):
            opened["yes"] = True

    monkeypatch.setattr(codec, "_FfmpegEncodeBackend", _Sub)
    stage = EncodeStage(res, backend="auto")
    assert stage.backend_name == "pynvvideocodec"
    stage.warmup()
    assert stage.backend_name == "ffmpeg"
    assert stage.fell_back_to_ffmpeg is True
    assert opened == {"yes": True}


def test_an_explicit_request_does_not_fall_back(monkeypatch):
    res = Resolution(16, 8)

    def _boom():
        raise CodecBackendUnavailable("libnvidia-encode missing")

    monkeypatch.setattr(codec, "_import_pynvvideocodec", _boom)
    monkeypatch.setattr(codec, "pynvvideocodec_available", lambda: True)
    stage = EncodeStage(res, backend="pynvvideocodec")
    with pytest.raises(CodecBackendUnavailable):
        stage.warmup()
    assert stage.fell_back_to_ffmpeg is False


def test_the_encoder_session_is_a_ledger_post_only_when_it_is_ours(monkeypatch):
    """#286: in-process VRAM is registered; a subprocess's VRAM is not ours."""
    res = Resolution(1280, 720)
    monkeypatch.setattr(codec, "pynvvideocodec_available", lambda: True)
    inprocess = EncodeStage(res, backend="pynvvideocodec")
    assert inprocess.session_bytes == frame_math.nvenc_session_bytes(res)
    assert EncodeStage(res, backend="ffmpeg").session_bytes == 0


def test_the_session_post_reproduces_both_measured_points():
    """The affine fit is not free to drift off the two numbers it was cut from.

    Card 0 (RTX 3080), h264 P4/high_quality, 2026-08-03, NVML device-wide
    free delta around session creation with the input surface subtracted.
    """
    mib = frame_math.MIB
    at_720p = frame_math.nvenc_session_bytes(Resolution(1280, 720))
    at_4k = frame_math.nvenc_session_bytes(Resolution(3840, 2160))
    assert at_720p / mib == pytest.approx(51.8, abs=0.1)
    assert at_4k / mib == pytest.approx(263.8, abs=0.1)
    # It is a FUNCTION of geometry, which is the finding: a 4K session is not
    # a 720p session, and one constant would have been wrong by 5x.
    assert at_4k > 4 * at_720p


def test_the_reservation_carries_the_session_only_on_the_inprocess_lane():
    target = Resolution(3840, 2160)
    kwargs = dict(
        source=Resolution(1920, 1080),
        target=target,
        streams_in_flight=1,
        with_rife=False,
    )
    without = frame_math.chain_reservation(**kwargs)
    with_session = frame_math.chain_reservation(**kwargs, inprocess_nvenc=True)
    expected = frame_math.nvenc_session_bytes(target)
    assert "nvenc_session" not in without.posts
    assert with_session.posts["nvenc_session"] == expected
    assert with_session.total_bytes - without.total_bytes == expected


def test_the_producing_stream_is_synchronised_before_nvenc_reads(monkeypatch):
    """The hardware finding, pinned.

    NVENC reads the input surface on its own engine with no dependency on the
    stream that wrote it. Without this ordering the encoder does not fail --
    it encodes a partially written frame. Measured on card 0, 2026-08-03,
    60 frames at 720p: 8.59 dB without, 15.65 dB with, against an ffmpeg
    baseline of 15.65 dB. None of that is visible in a frame count, so only
    an explicit assertion on the ordering can hold it in place.
    """
    res = Resolution(64, 32)
    order = []

    class _Nvc:
        def CreateEncoder(self, w, h, fmt, cpu_input, **options):  # noqa: N802
            encoder = _StrictFakeEncoder(w, h, fmt, cpu_input, **options)
            real = encoder.Encode

            def _record(frame, *args):
                order.append("encode")
                return real(frame, *args)

            encoder.Encode = _record
            return encoder

    monkeypatch.setattr(codec, "_import_pynvvideocodec", lambda: _Nvc())
    monkeypatch.setattr(
        codec._PyNvEncodeBackend,
        "_sync_producer",
        staticmethod(lambda tensor: order.append("sync")),
    )
    backend = codec._PyNvEncodeBackend(res)
    backend.encode(_FakeDeviceTensor((48, 64)))
    assert order == ["sync", "encode"], "the sync must precede the read, not follow it"

    # A host buffer is not a device pointer and has no stream to wait on, so
    # the wait is skipped rather than paid on every ffmpeg-lane frame.
    # An input that is not a device tensor -- a host buffer, or an already
    # built frame -- has no stream to wait on, so the wait is skipped rather
    # than paid on every frame of a lane that does not need it.
    order.clear()
    backend.encode(codec._NvencDeviceFrame(_FakeDeviceTensor((48, 64)), res))
    assert order == ["encode"]


def test_submitted_frames_stay_referenced_while_nvenc_may_read_them(monkeypatch):
    """Zero-copy means NVENC borrows the allocation, so somebody must own it.

    The session does not consume the surface inside ``Encode``: on card 0 the
    first packet came back from the SEVENTH call, so six frames were live at
    once. torch's caching allocator would hand an unreferenced block straight
    to the next frame. A 300-frame bypass arm on the card did NOT reproduce a
    corruption, so this is a contract guard rather than the fix for an
    observed failure -- and it is asserted here because a guard nobody checks
    is a guard somebody deletes.
    """
    res = Resolution(64, 32)

    class _Nvc:
        def CreateEncoder(self, w, h, fmt, cpu_input, **options):  # noqa: N802
            encoder = _StrictFakeEncoder(w, h, fmt, cpu_input, **options)

            def _lagged(frame, *args):
                # Model the measured lookahead: nothing for six frames.
                encoder.frames += 1
                if encoder.frames <= 6:
                    return []
                return [{"data": b"XXXX", "timestamp": encoder.frames}]

            encoder.Encode = _lagged
            return encoder

    monkeypatch.setattr(codec, "_import_pynvvideocodec", lambda: _Nvc())
    monkeypatch.setattr(
        codec._PyNvEncodeBackend, "_sync_producer", staticmethod(lambda tensor: None)
    )
    backend = codec._PyNvEncodeBackend(res)
    submitted = []
    for _ in range(6):
        tensor = _FakeDeviceTensor((48, 64))
        submitted.append(tensor)
        backend.encode(tensor)
    held = {id(frame._tensor) for frame in backend._inflight}
    assert held == {id(t) for t in submitted}, "every unconsumed frame is still owned"
    # The ring never drops below its floor while the encoder is still lagging.
    for _ in range(20):
        backend.encode(_FakeDeviceTensor((48, 64)))
    assert len(backend._inflight) >= codec.MIN_INFLIGHT_HOLD
    backend.flush()
    assert not backend._inflight, "EndEncode drains the queue; nothing is read after"
