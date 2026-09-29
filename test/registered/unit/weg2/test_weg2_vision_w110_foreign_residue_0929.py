"""W110 F10-Rest (29.09.): the teardown verdict judges the stage's OWN surviving
allocator blocks, not the process-wide delta.

METAL. NF z30w-park (0929_082743, 50ae2014b0, PP0 on nvml1), the one boot with
the torch figure::

    W102 Weg2VisionStage run=1 ... vram_residue_mib=-5136.0 torch_residue_mib=+6.5
    W110 Weg2VisionTeardownIncomplete run=1: ... 6.5 MiB more ... (torch allocated delta)

and the three NF 0928 vision boots (NVML basis): run=1 +20.0 / +18.0 / +20.0,
every later run +0.0 / -4 / -66 / -5070.

WHY THE OLD BASIS CANNOT JUDGE. Every tower tensor is released by
construction (parameters are views on the KV tail, buffers stripped, the rope
popped from ``_ROPE_DICT``, the rows moved to the HOST in ``encode_items``),
and ``torch.cuda.memory_allocated`` is ONE counter for the whole device and
every thread of PP0 (HiCache, the PP transport, and -- async form -- whole
forward passes inside the window). A live 1-10 MiB block is served from a
20 MiB large-buffer segment (PyTorch caching allocator, kLargeBuffer), which
``empty_cache`` cannot return while the block lives: the NVML +20 (+18 with a
2 MiB small segment given back) is one such segment, and +6.5 MiB torch is a
block of exactly that class.

THE FIX. The caching allocator's own snapshot before the build and after the
teardown: the blocks that survived, with their stream and segment. A block on
a stream the stage allocated on (the scheduler's current stream, the read
stream, the async worker's stream) is the stage's -> W110; any other is
foreign -> W110b, named, no verdict. Snapshot missing -> the torch delta as
before.

RED on 895559fed2 (no snapshot attribution: the foreign 6.5 MiB block fires
W110), GREEN with the fix. Hermetic, CPU: the snapshot is the real format of
``torch.cuda.memory_snapshot`` (keys pinned against
torch/csrc/cuda/Module.cpp: address, total_size, stream, segment_type, blocks
with address/size/state).
"""

import logging

import torch

from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs

MIB = vrs.MIB
SCHED = 0x0          # the scheduler thread's current stream (legacy default)
READ = 0x7F10        # the stage's read stream
FOREIGN = 0x5A20     # another thread's stream (HiCache / PP transport / forward_stream)
CUDA0 = torch.device("cuda", 0)


def _seg(addr, total, stream, blocks, kind="large", device=0):
    return {"device": device, "address": addr, "total_size": total, "stream": stream,
            "segment_type": kind,
            "blocks": [{"address": a, "size": s, "state": st} for a, s, st in blocks]}


BEFORE = [
    _seg(0x1000_0000, 20 * MIB, SCHED, [(0x1000_0000, 4 * MIB, "active_allocated"),
                                        (0x1000_0000 + 4 * MIB, 16 * MIB, "inactive")]),
    _seg(0x2000_0000, 2 * MIB, FOREIGN, [(0x2000_0000, 512, "active_allocated"),
                                          (0x2000_0200, 2 * MIB - 512, "inactive")], kind="small"),
]

#: z30w-park run=1: every tower block gone, one 6.5 MiB block another stream
#: allocated inside the window, in a 20 MiB segment that did not exist before
AFTER_FOREIGN = BEFORE + [
    _seg(0x3000_0000, 20 * MIB, FOREIGN, [(0x3000_0000, int(6.5 * MIB), "active_allocated"),
                                           (0x3000_0000 + int(6.5 * MIB), int(13.5 * MIB), "inactive")]),
]

#: a genuine teardown leak: the block sits on the stage's own stream
AFTER_OWN = BEFORE + [
    _seg(0x4000_0000, 20 * MIB, READ, [(0x4000_0000, int(6.5 * MIB), "active_allocated"),
                                        (0x4000_0000 + int(6.5 * MIB), int(13.5 * MIB), "inactive")]),
]


def _snap(segs):
    return lambda: segs


def _blocks(segs):
    return vrr.live_blocks(CUDA0, snapshot=_snap(segs))


def _outcome(after, torch_delta=int(6.5 * MIB), nvml=20 * MIB):
    out = vrr.StageOutcome()
    out.residue_bytes = nvml
    out.torch_residue_bytes = torch_delta
    vrr.note_survivors(out, vrr.stage_survivors(_blocks(BEFORE), _blocks(after), [SCHED, READ]))
    return out


def test_live_blocks_reads_the_real_snapshot_format():
    lb = _blocks(BEFORE + [_seg(0x9000_0000, 2 * MIB, SCHED,
                                [(0x9000_0000, 1024, "active_allocated")], device=1)])
    assert set(lb["blocks"]) == {0x1000_0000, 0x2000_0000}, "active only, this device only"
    assert lb["blocks"][0x1000_0000] == (4 * MIB, SCHED, 20 * MIB, "large", 0x1000_0000)
    assert lb["segments"] == {0x1000_0000, 0x2000_0000}


def test_z30w_park_shape_is_a_foreign_block_not_W110(caplog):
    caplog.set_level(logging.INFO)
    out = _outcome(AFTER_FOREIGN)
    (s,) = out.survivors
    assert (s.size, s.stream, s.own) == (int(6.5 * MIB), FOREIGN, False)
    assert s.segment_bytes == 20 * MIB and s.segment_type == "large" and s.new_segment, \
        "the 20 MiB large-buffer segment NVML saw as +20"
    assert out.own_residue_bytes == 0 and out.foreign_residue_bytes == int(6.5 * MIB)
    leak, basis = vrr.teardown_leak_bytes(out)
    assert leak == 0 and "own surviving blocks" in basis
    vrr.log_outcome(out, ["weg2-21-52"], 1)
    assert vrr.W_TEARDOWN + " run=1" not in caplog.text, "no TeardownIncomplete for a foreign block"
    assert vrr.W_FOREIGN_RESIDUE in caplog.text
    assert "6.5MiB@foreign:stream=0x5a20:seg=20MiB/large/new" in caplog.text
    assert "own_residue_mib=+0.0 foreign_residue_mib=+6.5" in caplog.text
    assert "torch_residue_mib=+6.5" in caplog.text, "the process-wide figure stays in the line"


def test_a_block_on_the_stages_own_stream_still_fires_W110_and_is_named(caplog):
    caplog.set_level(logging.INFO)
    out = _outcome(AFTER_OWN)
    leak, _ = vrr.teardown_leak_bytes(out)
    assert leak == int(6.5 * MIB) > vrr.RESIDUE_TOLERANCE_BYTES
    vrr.log_outcome(out, ["weg2-1-1"], 1)
    assert vrr.W_TEARDOWN + " run=1" in caplog.text
    assert "survivors=(6.5MiB@own:stream=0x7f10:seg=20MiB/large/new)" in caplog.text
    assert vrr.W_FOREIGN_RESIDUE not in caplog.text


def test_a_block_live_before_the_build_is_no_survivor():
    out = _outcome(BEFORE, torch_delta=0, nvml=0)
    assert out.survivors == [] and out.own_residue_bytes == 0 and out.foreign_residue_bytes == 0


def test_a_reused_address_with_a_new_size_is_a_survivor():
    after = [_seg(0x1000_0000, 20 * MIB, SCHED, [(0x1000_0000, 8 * MIB, "active_allocated")])]
    (s,) = vrr.stage_survivors(_blocks(BEFORE), _blocks(after), [SCHED])
    assert s.size == 8 * MIB and s.own and not s.new_segment


def test_without_a_snapshot_the_torch_delta_judges_as_before(caplog):
    out = vrr.StageOutcome()
    out.torch_residue_bytes = int(6.5 * MIB)
    vrr.note_survivors(out, vrr.stage_survivors(None, _blocks(AFTER_FOREIGN), [SCHED]))
    assert out.own_residue_bytes is None
    leak, basis = vrr.teardown_leak_bytes(out)
    assert leak == int(6.5 * MIB) and "torch" in basis


def test_a_snapshot_that_raises_is_unmeasured():
    def boom():
        raise RuntimeError("no allocator")
    assert vrr.live_blocks(CUDA0, snapshot=boom) is None
    assert vrr.live_blocks(torch.device("cpu")) is None


def test_both_stage_paths_take_the_snapshot_and_the_read_stream():
    import inspect
    sync = inspect.getsource(vrr.run_rank_stage)
    assert "live_blocks(device)" in sync and "own_streams.append(_stream_id(stream))" in sync
    assert "stage_survivors(blocks0" in sync
    a_start = inspect.getsource(vrr.start_async_stage)
    a_finish = inspect.getsource(vrr.finish_async_stage)
    assert "own_streams.append(_stream_id(stream))" in a_start
    assert "stage_survivors(st.blocks0" in a_finish
