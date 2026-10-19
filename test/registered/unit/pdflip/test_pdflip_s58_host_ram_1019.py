"""S58 (19.10., NF rc15 boot 1019_071429): host RAM nobody needs.

Measured 08:30-08:47Z on the running container (cgroup memory.current
102.8 GiB of 104 GiB, /proc/<pid>/smaps per rank):

* ``<arena>/handoff/`` held 565 ``pdflip-*.json`` hand-off files (318 MiB of
  /dev/shm) against ONE pending and ONE park mark -- the file outlived its rid
  (``read_keys(remove=False)`` since xsn330, the hold release removes only
  held rids). 63-84 files per 10 min, unbounded.
* D TP0 held two 2048 MiB ``/dev/zero`` blocks for the 1522.8 MiB draft image
  and the 1386.2 MiB tail-stage arena: torch's CachingHostAllocator rounds
  every pin to the next power of two. 1.16 GiB nobody uses.

Hermetic, CPU only: the allocators are recorded, the hand-off files are real
files in a temp arena directory.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

import flliper.srt.layers.moe.expert_offload as eo  # noqa: E402
from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import handoff as ho  # noqa: E402
from flliper.srt.pdflip import handoff_pending as hp  # noqa: E402
from flliper.srt.pdflip import tail_adopt as ta  # noqa: E402

TAIL_ARENA = 1453524096  # TAIL-STAGE-ARENA bytes= on D TP0, 1019_071429
DRAFT_IMAGE = 1596748576  # PDFLIP-DRAFT-PARK host image bytes= on D TP0


# -- the hand-off file ends with its rid -------------------------------------
@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_HANDOFF_REMOVE_AT_RID_END", raising=False)
    return tmp_path


def _publish(rid):
    assert ho.write(rid, [1, 2, 3, 4], ["aa", "bb"])
    assert os.path.exists(ho.path(rid))


def test_front_rid_end_removes_the_hand_off_file(arena, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    _publish("pdflip-0-1")
    hp.mark("pdflip-0-1", 2, 4)
    hp.drop("pdflip-0-1", "served")
    assert not os.path.exists(ho.path("pdflip-0-1")), "the rid is over: its hand-off file must go with it"


def test_rid_end_without_a_pending_mark_removes_it_too(arena):
    """The mark is consumed at D's prefetch termination, long before the rid
    ends -- the file must still go at the front's rid end (the measured case:
    565 files, one pending mark)."""
    _publish("pdflip-58-488")
    hp.drop("pdflip-58-488", "served")
    assert not os.path.exists(ho.path("pdflip-58-488"))


def test_d_consume_keeps_the_file_for_the_dormant_hold(arena, monkeypatch):
    """xsn330: the dormant hold re-reads the keys every 2 s after the prefetch
    terminated -- D's consume must not take the file."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    _publish("pdflip-3-9")
    hp.consume("pdflip-3-9", "fetch")
    assert os.path.exists(ho.path("pdflip-3-9"))
    assert ho.read_keys("pdflip-3-9", remove=False) == ["aa", "bb"]


def test_switch_off_keeps_the_file(arena, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF_REMOVE_AT_RID_END", "0")
    _publish("pdflip-1-2")
    hp.drop("pdflip-1-2", "served")
    assert os.path.exists(ho.path("pdflip-1-2"))


def test_rid_end_of_a_rid_without_a_file_is_quiet(arena):
    hp.drop("pdflip-9-99", "disconnect")  # served on D only: nothing was written
    assert not os.path.exists(ho.path("pdflip-9-99"))


def test_other_rids_files_stay(arena):
    _publish("pdflip-4-5")
    _publish("pdflip-4-6")
    hp.drop("pdflip-4-5", "served")
    assert not os.path.exists(ho.path("pdflip-4-5"))
    assert os.path.exists(ho.path("pdflip-4-6"))


# -- exact-size pins instead of power-of-two blocks --------------------------
class _Rec:
    def __init__(self):
        self.exact, self.pinned = [], []


@pytest.fixture
def rec(monkeypatch):
    r = _Rec()
    real_empty = torch.empty

    def fake_exact(shape, dtype):
        r.exact.append((tuple(shape), dtype))
        return real_empty(64, dtype=dtype)

    def fake_empty(*args, **kw):
        if kw.get("pin_memory"):
            r.pinned.append(args[0])
            return real_empty(64, **{k: v for k, v in kw.items() if k != "pin_memory"})
        return real_empty(*args, **kw)

    monkeypatch.setattr(eo, "pinned_exact_empty", fake_exact)
    monkeypatch.setattr(ta.torch, "empty", fake_empty)
    return r


def test_tail_stage_arena_is_pinned_at_its_exact_size(rec):
    ta.PinArena(TAIL_ARENA, pin=True)
    assert rec.exact == [((TAIL_ARENA,), torch.uint8)]
    assert rec.pinned == [], "torch.empty(pin_memory=True) rounds 1386.2 MiB to a 2048 MiB block"


def test_tail_stage_exact_pin_switch_off_is_the_torch_form(rec):
    with envs.FLLIPER_OPT_PDFLIP_TAIL_STAGE_EXACT_PIN.override(False):
        ta.PinArena(TAIL_ARENA, pin=True)
    assert rec.exact == [] and rec.pinned == [TAIL_ARENA]


def test_tail_stage_unpinned_arena_never_takes_the_exact_path(rec):
    a = ta.PinArena(4096, pin=False)
    assert rec.exact == [] and rec.pinned == []
    assert a.take("r", (16,), torch.float32) is not None


def test_tail_stage_failed_registration_falls_back(rec, monkeypatch):
    def boom(shape, dtype):
        raise RuntimeError("cudaHostRegister failed with cudaError 1")

    monkeypatch.setattr(eo, "pinned_exact_empty", boom)
    a = ta.PinArena(TAIL_ARENA, pin=True)
    assert rec.pinned == [TAIL_ARENA] and a.buf is not None


def test_tail_stage_arena_spans_work_on_the_exact_buffer(monkeypatch):
    """take/release are views into the one buffer, whatever allocated it."""
    monkeypatch.setattr(eo, "pinned_exact_empty", lambda shape, dtype: torch.zeros(shape, dtype=dtype))
    a = ta.PinArena(1 << 16, pin=True)
    t = a.take("pdflip-1-1", (8, 4), torch.bfloat16)
    assert t is not None and t.shape == (8, 4)
    assert a.release("pdflip-1-1") > 0


def test_draft_image_exact_pin_is_on_by_default():
    assert envs.FLLIPER_OPT_PDFLIP_DRAFT_PARK_EXACT_PIN.get() is True, (
        "user order 14.10.: new functions default ON; off it held 2048 MiB for a 1522.8 MiB image")


def test_draft_image_default_goes_through_the_exact_helper(monkeypatch):
    from flliper.srt.pdflip import draft_park as dp

    seen = []
    monkeypatch.setattr(eo, "pinned_exact_empty", lambda shape, dtype: seen.append(tuple(shape)) or torch.empty(8, dtype=dtype))
    dp._host_image(DRAFT_IMAGE, pin=True)
    assert seen == [(DRAFT_IMAGE,)]


def test_draft_image_failed_registration_falls_back(monkeypatch):
    from flliper.srt.pdflip import draft_park as dp

    def boom(shape, dtype):
        raise RuntimeError("cudaHostRegister failed with cudaError 1")

    pinned = []
    real_empty = torch.empty

    def fake_empty(*args, **kw):
        if kw.get("pin_memory"):
            pinned.append(args[0])
            return real_empty(8, dtype=kw.get("dtype"))
        return real_empty(*args, **kw)

    monkeypatch.setattr(eo, "pinned_exact_empty", boom)
    monkeypatch.setattr(dp.torch, "empty", fake_empty)
    dp._host_image(DRAFT_IMAGE, pin=True)
    assert pinned == [DRAFT_IMAGE]
