"""L15-12 part 2 (AP, 1001): the D WAKE's group fence carries each rank's
manifest fingerprint (``l15_fp``), reduced group-wide by the pure
``l15_restore.l15_fp_reduce``; the D wake then decides hold/fallback from the
group fingerprint (master-gated: off = byte-identical).

Step 1 (this commit) tests the pure reduce helper + its verdict tie-in:
  * all-None votes  -> (None, False)            (nobody holds -> "none")
  * equal ints       -> (x, x), mixed False     -> verdict "hold"
  * unequal ints     -> (min, max), mixed False -> verdict "fallback"
  * int + None mix   -> mixed True              (the wake site forces fallback)

The wake-site wiring (master-gated ``l15_fp`` on the resume fence) is covered
by a source check appended in L15-12 step 3 once the wake site lands.

Hermetic: pure-Python helper only (no torch/CUDA import).
"""

import sys
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

from flliper.srt.pdflip import l15_restore  # noqa: E402


def test_fp_reduce_all_none_is_none():
    # Nobody held anything (master off, or all ranks fell back to flush):
    # the group verdict must be "none" -- the wake path must NOT touch the
    # KV (today's byte-identical behaviour).
    mm, mixed = l15_restore.l15_fp_reduce([None, None])
    assert mm is None and mixed is False
    assert l15_restore.verdict(None, None, None) == "none"


def test_fp_reduce_equal_is_hold():
    # Identical hold across the group -> the group keeps the hold.
    mm, mixed = l15_restore.l15_fp_reduce([7, 7, 7])
    assert mm == (7, 7) and mixed is False
    assert l15_restore.verdict(7, *mm) == "hold"


def test_fp_reduce_unequal_is_fallback():
    # Ranks hold different content -> group must fall back together.
    mm, mixed = l15_restore.l15_fp_reduce([7, 8])
    assert mm == (7, 8) and mixed is False
    assert l15_restore.verdict(7, *mm) == "fallback"


def test_fp_reduce_mixed_flag():
    # Some ranks held, some did not -> mixed; the wake site maps mixed to
    # the group-wide fallback (the int-bearing ranks must drop their hold too,
    # or the KV state diverges across the group).
    mm, mixed = l15_restore.l15_fp_reduce([7, None, 7])
    assert mm == (7, 7) and mixed is True
    # And the all-None vs int boundary: a lone holder in a group of Nones.
    assert l15_restore.l15_fp_reduce([None, 5]) == ((5, 5), True)


def test_fp_reduce_empty_and_single():
    assert l15_restore.l15_fp_reduce([]) == (None, False)
    assert l15_restore.l15_fp_reduce([42]) == ((42, 42), False)


def test_resume_fence_passes_l15_fp_only_behind_master_gate():
    """Source check (L15-12 part 2): the D wake's resume fence carries
    ``l15_fp``, and the fingerprint is read only behind the D-group +
    ``master_on`` gate -- so with the master off ``_l15_fp`` stays None and
    the fence's gather payload is byte-identical to before."""
    import re

    text = (REPO / "python" / "flliper" / "srt" / "managers"
            / "scheduler_components" / "weight_updater.py").read_text()
    m = re.search(r"def resume_memory_occupation", text)
    assert m, "def resume_memory_occupation not found"
    nxt = text.find("\n    def ", m.end())
    assert nxt != -1
    src = text[m.start():nxt]

    # The resume fence call passes the kwarg...
    fence = src.find("l15_fp=_l15_fp")
    assert fence != -1, "resume fence does not pass l15_fp"
    # ...but _l15_fp is initialised to None (the byte-identical default)...
    init = src.find("_l15_fp: Optional[int] = None")
    assert init != -1, "_l15_fp has no None default"
    # ...and is only assigned a non-None value under the group + master gate
    # (the combined condition line is the wake-site's own; the group name is
    # also compared elsewhere in the method).
    grp = src.find('pdflip_memory_saver_on and self._pdflip_group_name() == "D"')
    gate = src.find("l15_plan.master_on(os.environ)")
    # F11 (01.10.): the fence read moved into _l15_fence_manifest -- the
    # wake restore already read-and-unlinked the record, so the fence takes
    # the stashed self._l15_wake_manifest and only falls back to
    # load_for_wake when it is not set.  The helper must keep BOTH sides.
    read = src.find("self._l15_fence_manifest(")
    assign = src.find("_l15_fp = int(")
    for name, pos in (("D-group gate", grp), ("master_on gate", gate),
                      ("fence manifest read", read),
                      ("fingerprint assign", assign)):
        assert pos != -1, f"missing: {name}"
    assert init < grp < gate < read < assign < fence, (
        "the manifest read/assign must sit behind the D-group and "
        "master_on gates, and the fence call after the assign")
    helper = text[text.find("def _l15_fence_manifest"):]
    helper = helper[:helper.find("\n    def ")]
    assert helper, "def _l15_fence_manifest not found"
    assert "m = self._l15_wake_manifest" in helper, (
        "fence manifest read does not take the stashed record first")
    assert "load_for_wake(manifest_path)" in helper, (
        "fence manifest read does not fall back to the file")


def _three_rank_manifest():
    from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

    return Manifest(
        epoch=1,
        pid=4242,
        spans=(HoldSpan(
            rid="a", depth=3, slots=(0, 1, 2), anchor_slot=0,
            l2_slots=(10, 11, 12), l2_gens=(1, 1, 1)),),
        rows_by_rank=(4, 4, 4),
        anchor_slots=0,
    )


def test_refill_plan_uses_planner_caps_not_manifest_capacity():
    """The D3 defect: the manifest's rows_by_rank is the per-rank KEEP
    CAPACITY (blocks * ratio), > 0 for TP0 too; fed as cap_rows_by_rank it
    would make refill_plan return [] for EVERY rank, TP0 included -- the one
    rank that must refill from L2.  The cap signal is the planner's per-card
    cap (caps_from_env): (0, 100, 100) -> rank 0 plans its owned rows, the
    kept ranks plan []."""
    m = _three_rank_manifest()
    prefix = [0, 1, 2, 3]
    assert l15_restore.refill_plan(m, 0, prefix, (0, 100, 100)) == [(0, 10, 1)]
    assert l15_restore.refill_plan(m, 1, prefix, (0, 100, 100)) == []
    assert l15_restore.refill_plan(m, 2, prefix, (0, 100, 100)) == []
    # And the pinned anti-shape: the manifest capacity disables every rank.
    assert l15_restore.refill_plan(m, 0, prefix, m.rows_by_rank) == []


def test_wake_site_feeds_planner_caps_into_refill_plan():
    """Source check (D3 review): the wake site's refill_plan call is fed
    the planner caps (caps_from_env), never the manifest's rows_by_rank."""
    import re

    text = (REPO / "python" / "flliper" / "srt" / "managers"
            / "scheduler_components" / "weight_updater.py").read_text()
    m = re.search(r"def resume_memory_occupation", text)
    assert m, "def resume_memory_occupation not found"
    nxt = text.find("\n    def ", m.end())
    assert nxt != -1
    src = text[m.start():nxt]

    call = src.find("l15_restore.refill_plan(")
    assert call != -1, "no refill_plan call at the wake site"
    close = src.find("_l15_missing", call)
    assert close != -1
    args = src[call:close]
    assert "rows_by_rank" not in args, (
        "the wake refill plan is still fed the manifest keep capacity "
        "(rows_by_rank) instead of the planner caps")
    assert "_l15_caps" in args
    caps = src.find("caps_from_env(")
    assert caps != -1 and caps < call, (
        "the wake site does not derive its caps via l15_shadow.caps_from_env")