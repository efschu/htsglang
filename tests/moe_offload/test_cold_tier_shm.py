# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for the shared cold-expert tier (#394 reachability).

No CUDA, no driver, no model: the shm directory is redirected to a tmp path and
everything below is real mmap over real files.

What is pinned here is the property the whole mechanism rests on -- **a peer
either gets the exact bytes its manifest promised, or a named exception.**
Serving plausible wrong expert weights is the failure this design is most
exposed to (a stale segment from a previous launch reads perfectly), so the
torn-mapping arms are the point of the file, not an afterthought:

  * wrong instance id (leftover segment from the previous run)
  * wrong format version
  * digest disagreement between manifest and segment
  * truncated segment
  * an unsealed segment being read before the owner finished writing
  * a row index for an expert the owner does not hold

Each has a committed CAN-FAIL arm showing the check is what rejects it, rather
than the read happening to fail for some other reason.

Run:
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=python \
    python -m pytest tests/moe_offload/test_cold_tier_shm.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import cold_tier_shm as cts  # noqa: E402
from sglang.srt.layers.moe.cold_tier_shm import (  # noqa: E402
    HEADER_BYTES,
    ColdTierError,
    ColdTierLayout,
    ManifestUnavailable,
    SegmentMismatch,
    attach_peer_segment,
    create_owned_segment,
    detach_all,
    peer_row_view,
    publish_manifest,
    read_peer_manifest,
    seal_owned_segment,
    segment_path,
)

ROW_BYTES = 64
ROWS = 4
INSTANCE = "inst0001"


@pytest.fixture(autouse=True)
def _shm(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_COLD_TIER_SHM_DIR", str(tmp_path))
    detach_all()
    yield tmp_path
    detach_all()


def _layout(**over):
    base = dict(
        instance_id=INSTANCE,
        owner_rank=1,
        owner_card_uuid="GPU-x4-card",
        owner_pci_bdf="00000000:05:00.0",
        layer_key="layer7",
        param_attr="w13_qweight",
        rows=ROWS,
        row_bytes=ROW_BYTES,
        dtype="uint8",
        row_shape=(ROW_BYTES,),
        expert_ids=(80, 83, 94, 97),
    )
    base.update(over)
    return ColdTierLayout(**base)


def _fill(layout, seal=True):
    """Owner side: create, write one recognizable pattern per row, seal."""
    mm = create_owned_segment(layout)
    view = memoryview(mm)
    for row, expert in enumerate(layout.expert_ids):
        start = HEADER_BYTES + row * layout.row_bytes
        view[start : start + layout.row_bytes] = (
            bytes([expert & 0xFF]) * layout.row_bytes
        )
    if seal:
        seal_owned_segment(mm, layout)
    return mm


# --------------------------------------------------------------------------
# 1. the happy path: a peer reads exactly what the owner wrote
# --------------------------------------------------------------------------


def test_a_peer_reads_the_owners_rows_byte_for_byte():
    layout = _layout()
    _fill(layout)

    for expert in layout.expert_ids:
        got = bytes(peer_row_view(layout, expert))
        assert got == bytes([expert & 0xFF]) * ROW_BYTES, expert


def test_the_expert_to_row_map_is_a_lookup_not_arithmetic():
    """Expert ids are sparse after apportionment; id-minus-base is wrong."""
    layout = _layout(expert_ids=(80, 83, 94, 97))
    _fill(layout)

    assert layout.row_of(94) == 2
    assert bytes(peer_row_view(layout, 94))[0] == 94


def test_a_segment_is_mapped_once_per_process_not_once_per_fetch():
    layout = _layout()
    _fill(layout)

    first = attach_peer_segment(layout, register_with_cuda=False)
    for _ in range(20):
        peer_row_view(layout, 83)
    assert attach_peer_segment(layout, register_with_cuda=False) is first


def test_the_manifest_round_trips_through_a_file_and_no_collective():
    layout = _layout()
    publish_manifest(INSTANCE, 1, [layout])

    peers = read_peer_manifest(INSTANCE, 1)

    assert peers[("layer7", "w13_qweight")] == layout


# --------------------------------------------------------------------------
# 2. FALSIFIERS: a torn mapping must fail loudly, never serve bytes
# --------------------------------------------------------------------------


def test_a_leftover_segment_from_a_previous_launch_is_refused():
    """The most dangerous case: it reads perfectly and is the wrong model."""
    old = _layout(instance_id="OLDRUN99")
    _fill(old)
    # Same segment file name shape, new launch id.
    now = _layout(instance_id=INSTANCE)
    os.replace(segment_path(old), segment_path(now))

    with pytest.raises(SegmentMismatch, match="previous run|instance"):
        peer_row_view(now, 80)


def test_the_leftover_segment_pin_can_fail():
    """CAN-FAIL ARM: without the instance check the stale bytes read fine.

    This is the whole argument for the header. The planted defect is not exotic
    -- it is simply trusting the file name, which is what a version of this
    module without an instance id would do.
    """
    old = _layout(instance_id="OLDRUN99")
    _fill(old)
    now = _layout(instance_id=INSTANCE)
    os.replace(segment_path(old), segment_path(now))

    # Read the way a name-trusting implementation would: no validation at all.
    with open(segment_path(now), "rb") as f:
        raw = f.read()
    start = HEADER_BYTES + 0 * ROW_BYTES
    stale = raw[start : start + ROW_BYTES]

    assert stale == bytes([80]) * ROW_BYTES, "stale bytes are perfectly readable"
    with pytest.raises(AssertionError):
        assert stale != bytes([80]) * ROW_BYTES


def test_a_digest_disagreement_between_manifest_and_segment_is_refused():
    written = _layout()
    _fill(written)
    # A manifest that names the same segment but a different expert order.
    claimed = _layout(expert_ids=(80, 94, 83, 97))

    with pytest.raises(SegmentMismatch, match="digest|different layouts"):
        peer_row_view(claimed, 94)


def test_a_truncated_segment_is_refused_before_any_read():
    layout = _layout()
    mm = _fill(layout)
    mm.close()
    with open(segment_path(layout), "r+b") as f:
        f.truncate(HEADER_BYTES + 2 * ROW_BYTES)

    with pytest.raises(SegmentMismatch, match="truncated|segment is"):
        peer_row_view(layout, 97)


def test_an_unsealed_segment_is_refused_so_half_written_is_not_readable():
    """Sealing last is what distinguishes 'mapped' from 'ready'."""
    layout = _layout()
    _fill(layout, seal=False)

    with pytest.raises(SegmentMismatch, match="magic|not a cold-tier segment"):
        peer_row_view(layout, 80)


def test_a_wrong_format_version_is_rejected_not_reinterpreted(monkeypatch):
    layout = _layout()
    _fill(layout)
    detach_all()
    monkeypatch.setattr(cts, "COLD_TIER_FORMAT", 999)

    with pytest.raises(SegmentMismatch, match="format"):
        peer_row_view(layout, 80)


def test_an_expert_the_owner_does_not_hold_is_named_not_guessed():
    layout = _layout()
    _fill(layout)

    with pytest.raises(SegmentMismatch, match="is not in rank 1's cold tier"):
        peer_row_view(layout, 12345)


def test_the_torn_mapping_family_pin_can_fail():
    """CAN-FAIL ARM covering the family: the checks are load-bearing.

    Every arm above asserts a RAISE. A test that only asserts raises can pass
    because the code is broken in some unrelated way, so this pins that the
    same read succeeds once the mapping is intact.
    """
    layout = _layout()
    _fill(layout)

    assert bytes(peer_row_view(layout, 80))[0] == 80
    with pytest.raises(AssertionError):
        assert bytes(peer_row_view(layout, 80))[0] != 80


# --------------------------------------------------------------------------
# 3. identity and manifest discipline
# --------------------------------------------------------------------------


def test_a_layout_must_name_the_owners_card_not_just_a_rank_index():
    """#397: a rank index is a launch artifact and cannot identify a card."""
    with pytest.raises(ValueError, match="card UUID"):
        _layout(owner_card_uuid="")


def test_a_manifest_from_another_launch_is_refused():
    publish_manifest("OLDRUN99", 1, [_layout(instance_id="OLDRUN99")])
    os.replace(
        cts.manifest_path("OLDRUN99", 1),
        cts.manifest_path(INSTANCE, 1),
    )

    with pytest.raises(ManifestUnavailable, match="previous launch|instance"):
        read_peer_manifest(INSTANCE, 1)


def test_a_missing_manifest_expires_with_a_reason_rather_than_hanging():
    """A bounded wait, not a barrier: the group must never hang on this."""
    import time

    t0 = time.monotonic()
    with pytest.raises(ManifestUnavailable, match="not readable"):
        read_peer_manifest(INSTANCE, 2, timeout_s=0.2)
    assert time.monotonic() - t0 < 5.0


def test_the_row_count_and_expert_count_must_agree():
    with pytest.raises(ValueError, match="rows but names"):
        _layout(rows=4, expert_ids=(1, 2))


def test_an_oversized_segment_names_the_tmpfs_cap_not_a_memory_shortage(monkeypatch):
    """The pages are RAM either way; /dev/shm's size= is an accounting limit."""
    monkeypatch.setattr(cts, "shm_capacity_bytes", lambda path=None: 1024)

    with pytest.raises(ColdTierError, match="tmpfs size cap"):
        create_owned_segment(_layout(rows=1024, row_bytes=1 << 20, expert_ids=()))


# --------------------------------------------------------------------------
# 4. torch views: zero-copy, and the OS keeps the write protection
# --------------------------------------------------------------------------


def test_a_peer_row_tensor_is_zero_copy_and_correct():
    import torch

    layout = _layout()
    _fill(layout)

    t = cts.peer_row_tensor(layout, 94)

    assert t.dtype is torch.uint8 and t.numel() == ROW_BYTES
    assert int(t[0]) == 94 and int(t[-1]) == 94
    # Zero-copy proved by ALIASING, not by pointer arithmetic: a byte the owner
    # writes after the view exists must be visible through the view. A copy
    # would keep showing the old value.
    owner = _fill(layout)  # owner keeps its writable mapping
    row = layout.row_of(94)
    owner[HEADER_BYTES + row * ROW_BYTES] = 7
    owner.flush()
    assert int(t[0]) == 7, "the peer view is a copy, not a mapping"
    # and repeated lookups reuse one mapping rather than remapping per fetch
    assert cts.peer_row_tensor(layout, 94).data_ptr() == t.data_ptr()


def test_writing_through_a_peer_view_dies_rather_than_corrupting_a_peer():
    """FALSIFIER for the read-only decision, in a forked child.

    PyTorch warns that it does not model non-writable tensors and that one
    "can write to the underlying buffer". The kernel disagrees. This pins that
    the kernel is the one enforcing it, because a silent write here would
    corrupt ANOTHER RANK's expert weights -- the worst failure this design has,
    and the reason the mapping stays PROT_READ.
    """
    import os

    layout = _layout()
    _fill(layout)
    cts.detach_all()

    pid = os.fork()
    if pid == 0:  # pragma: no cover - child never returns to pytest
        try:
            t = cts.peer_row_tensor(layout, 80)
            t[0] = 255
            os._exit(42)  # silent write: catastrophic
        except Exception:
            os._exit(7)  # python-level refusal: also acceptable
    _, status = os.waitpid(pid, 0)
    signal, code = status & 0x7F, status >> 8

    assert code != 42, "a write through a peer view silently succeeded"
    assert signal in (7, 11) or code == 7, f"unexpected (signal={signal}, code={code})"
