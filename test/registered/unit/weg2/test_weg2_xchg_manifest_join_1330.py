# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- the manifest write-along, the cross-group join, the leg knob.

THE MEASURED FORM THIS IS RED AGAINST (boot weg2xsn20, log
``/spinning/evidence-665-f1/boot_weg2_weg2xsn20_3267f109fb_0911_154706.P.log``,
counted with ``log_grep``: bare 34 / genuine 34 / prose 0)::

    hook=source        is_source=1 descs=515 src_resolved=515/515 dst_resolved=0/515
    hook=authoritative is_source=0 descs=505 src_resolved=0/505   dst_resolved=505/505
    hook=destination   is_source=0 descs=231 src_resolved=0/231   dst_resolved=231/231

All 24 injection legs read ``verdict=NO-COMPARE`` under
``W74 Weg2XchgSourceMissing ... has no source pointer``.  Root, verified at
``15cf96c7ab``: ``weight_exchange_shadow.ptr_of`` (:3336-3344) answers only for
THIS rank of THIS group, and the hook decides which group is asked for the
source (:3329-3331); on ``destination``/``authoritative`` that is the PEER, so
``weight_exchange.py:1811`` writes ``src_ptr=None`` on every descriptor.

RED ON 15cf96c7ab by name: ``sglang.srt.weg2.xchg_manifest`` does not exist;
``wx.xchg_legs`` / ``wx.leg_direction`` / ``wx.leg_enabled`` /
``wx.legs_skipped_line`` do not exist (``code_search 'weg2-xchg-legs'
mode=branches`` = 0 files over 924 branches).
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

# The real geometries of the form this slice exists for.
P_CUT = (44, 10, 10)          # --pp-layer-set: group P, PP3
D_RANKS = 3                   # group D, TP3
CARDS = (0, 1, 2)
N_LAYERS = sum(P_CUT)
TAG = "weights_0"


def _stage_of_layer(layer: int) -> int:
    acc = 0
    for stage, n in enumerate(P_CUT):
        acc += n
        if layer < acc:
            return stage
    raise AssertionError(layer)


def _split(total, parts):
    """An UNEVEN split whose parts sum to total -- 17:7:8 by construction."""
    weights = (17, 7, 8)
    base = [total * w // sum(weights) for w in weights]
    base[-1] += total - sum(base)
    assert sum(base) == total and len(base) == parts
    return tuple(base)


def _piece(name, rows, cols, item=1, tag=TAG, cls=None):
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    return xm.ManifestPiece(param_name=name,
                            tensor_class=cls or sh.tensor_class(name),
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=tag, nbytes=rows * cols * item)


def _names():
    """One row-parallel and one column-parallel class per layer, INT8 shaped."""
    for i in range(N_LAYERS):
        yield f"model.layers.{i}.self_attn.qkv_proj.weight", 1536, 512, "rows"
        yield f"model.layers.{i}.mlp.down_proj.weight", 2048, 1024, "cols"


def _manifests(*, sharded=True, d_ranks=D_RANKS):
    """The six rows of one boot: P (whole, per stage) and D (cut, per rank)."""
    p_pieces = {r: [] for r in range(len(P_CUT))}
    d_pieces = {r: [] for r in range(d_ranks)}
    for name, rows, cols, axis in _names():
        layer = int(name.split(".")[2])
        stage = _stage_of_layer(layer)
        p_pieces[stage].append(_piece(name, rows, cols))
        if not sharded:
            for r in range(d_ranks):
                d_pieces[r].append(_piece(name, rows, cols))
            continue
        if axis == "rows":
            for r, w in enumerate(_split(rows, d_ranks)):
                d_pieces[r].append(_piece(name, w, cols))
        else:
            for r, w in enumerate(_split(cols, d_ranks)):
                d_pieces[r].append(_piece(name, rows, w))
    out = []
    for r, pieces in p_pieces.items():
        out.append(xm.RankManifest(group="P", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="b1",
                                   pieces=tuple(pieces)))
    for r, pieces in d_pieces.items():
        out.append(xm.RankManifest(group="D", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="b1",
                                   pieces=tuple(pieces)))
    return out


def _join(**kw):
    return xm.join_manifests(_manifests(**kw), pp_group="P", tp_group="D")


# ---------------------------------------------------------------------------
# (1) THE DIRECTION KNOB
# ---------------------------------------------------------------------------


def test_the_direction_names_are_the_canonical_ones_not_a_third_spelling():
    """``pp_to_tp``/``tp_to_pp`` already exist (phase_flip_plan.py:43-44).

    Asserted BY IDENTITY against the module that owns them, so a copy-pasted
    literal here would not satisfy it: two vocabularies for one fact diverge on
    the first rename, and ``seam_coverage.py:228`` already reads those.
    """
    from sglang.srt.layers.dcp import phase_flip_plan as pf

    assert wx.LEGS_PP_TO_TP == pf.PP_TO_TP
    assert wx.LEGS_TP_TO_PP == pf.TP_TO_PP
    assert wx.XCHG_LEGS_CHOICES == (
        wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP, wx.LEGS_BOTH)


def test_the_default_is_both_and_a_typo_never_silences_a_direction(monkeypatch):
    """An absent OR unrecognised value is ``both`` -- today's behaviour.

    The danger direction is the other one: a typo that turned a direction OFF
    would remove half a boot's legs and read as a lane that ran clean.
    """
    monkeypatch.delenv(wx.XCHG_LEGS_ENV, raising=False)
    assert wx.xchg_legs() == wx.LEGS_BOTH
    monkeypatch.setenv(wx.XCHG_LEGS_ENV, "  PP_to_TP ")
    assert wx.xchg_legs() == wx.LEGS_PP_TO_TP
    monkeypatch.setenv(wx.XCHG_LEGS_ENV, "pp-to-tp")     # a plausible typo
    assert wx.xchg_legs() == wx.LEGS_BOTH


@pytest.mark.parametrize("hook,group,expect", [
    ("source", "P", wx.LEGS_PP_TO_TP),
    ("destination", "D", wx.LEGS_PP_TO_TP),
    ("authoritative", "D", wx.LEGS_PP_TO_TP),
    ("source", "D", wx.LEGS_TP_TO_PP),
    ("destination", "P", wx.LEGS_TP_TO_PP),
    ("authoritative", "P", wx.LEGS_TP_TO_PP),
])
def test_the_direction_needs_no_new_plumbing(hook, group, expect):
    """``(hook, group)`` already decides it: ``source`` exports, the rest import."""
    assert wx.leg_direction(hook, group) == expect


def test_a_skipped_leg_is_named_with_its_count_and_both_directions(monkeypatch):
    monkeypatch.setenv(wx.XCHG_LEGS_ENV, wx.LEGS_PP_TO_TP)
    assert wx.leg_enabled("source", "P") is True
    assert wx.leg_enabled("destination", "D") is True
    assert wx.leg_enabled("source", "D") is False
    assert wx.leg_enabled("destination", "P") is False
    line = wx.legs_skipped_line(2, hook="source", group="D")
    assert "legs_skipped=2" in line
    assert "reason=direction-knob" in line
    assert f"leg_direction={wx.LEGS_TP_TO_PP}" in line
    assert f"armed={wx.LEGS_PP_TO_TP}" in line


def test_the_skip_counter_is_the_modules_because_the_mixin_has_slots():
    """``SchedulerWeightUpdaterManager`` is ``@dataclass(kw_only=True,
    slots=True)`` (weight_updater.py:307-308).

    A ``self._weg2_legs_skipped = ...`` inside the flip leg would raise
    AttributeError -- #1329's exact shape, a first write that raised on a slots
    dataclass and cost three boots.  This pins the counter's home.
    """
    import ast
    import inspect
    import textwrap

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    wx.reset_legs_skipped()
    assert wx.record_leg_skipped() == 1
    assert wx.record_leg_skipped() == 2
    assert wx.legs_skipped_total() == 2
    wx.reset_legs_skipped()

    src = inspect.getsource(
        wu.SchedulerWeightUpdaterManager._weg2_shadow_hook)
    assert "record_leg_skipped()" in src
    # OVER THE AST AGAIN, for the reason the write-site test states: the
    # production comment NAMES the forbidden write in order to explain why it
    # is forbidden, and a text grep cannot tell the explanation from the
    # defect.  Second instance of the #995 prose trap inside this one file.
    body = ast.parse(textwrap.dedent(src))
    written = {
        t.attr
        for node in ast.walk(body) if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
        and t.value.id == "self"
    }
    assert not written, (
        f"the hook assigns {sorted(written)} on self, and the manager is a "
        f"slots=True dataclass -- that raises AttributeError inside the flip "
        f"leg, which is #1329's own shape")


def test_the_launcher_publishes_the_knob_to_both_groups_and_pops_it():
    """Launcher OUTPUT, so an inherited shell value can never silence a leg."""
    import inspect

    from sglang.srt.weg2 import launcher

    pub = inspect.getsource(launcher.prepare_xchg_env)
    assert "XCHG_LEGS_ENV" in pub
    assert "legs" in inspect.signature(launcher.prepare_xchg_env).parameters
    env = inspect.getsource(launcher.build_env)
    assert "XCHG_LEGS_ENV" in env, (
        "the knob must be POPPED by build_env: an inherited "
        "SGLANG_WEG2_XCHG_LEGS=pp_to_tp would silently halve a boot's legs")


# ---------------------------------------------------------------------------
# (2) THE WRITE-ALONG
# ---------------------------------------------------------------------------


def test_the_manifest_keys_on_the_published_identity_not_a_private_tuple():
    """``manifest_entry`` (weight_exchange_shadow.py:2125) -- one identity.

    The same tuple ``card_manifest_entries`` (:2918) publishes and
    ``seam_digest`` (:474) keys on.  A private key here would be a fourth
    reading of one fact.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    p = _piece("model.layers.0.mlp.down_proj.weight", 2048, 1024)
    assert p.key == sh.manifest_entry(p.param_name, p.tensor_class,
                                      p.rows_full, p.cols_full, p.itemsize)


def test_a_manifest_round_trips_through_the_shared_dump_directory(tmp_path):
    man = _manifests()[0]
    path = xm.write_rank_manifest(man, str(tmp_path))
    assert os.path.basename(path) == xm.manifest_filename(
        man.rank, man.group, man.region_tag)
    assert man.region_tag in os.path.basename(path), (
        "the region tag must be IN the name: one rank runs two runners (main "
        "model and drafter) through the same write site, and without it the "
        "second clobbers the first")
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")], (
        "the write must be atomic: a half-written file makes the join refuse "
        "a tensor that IS placed, which is the false-red direction")
    back = xm.load_manifests(str(tmp_path))
    assert len(back) == 1 and back[0] == man


def test_the_group_is_in_the_file_name_so_p_cannot_overwrite_d(tmp_path):
    """#1292 paid for this once: P (booted second) overwrote D's dump."""
    for man in _manifests():
        xm.write_rank_manifest(man, str(tmp_path))
    assert len(xm.load_manifests(str(tmp_path))) == len(P_CUT) + D_RANKS


def test_a_stale_boots_manifest_is_filtered_not_joined(tmp_path):
    """A file from a previous boot in the same evidence directory is a WRONG
    answer, not a missing one -- it would plan extents no rank holds."""
    for man in _manifests():
        xm.write_rank_manifest(man, str(tmp_path))
    stale = _manifests()[0]
    xm.write_rank_manifest(
        xm.RankManifest(group="P", rank=99, card=0, region_tag=TAG,
                        boot_token="OLD", pieces=stale.pieces), str(tmp_path))
    assert all(m.boot_token == "b1"
               for m in xm.load_manifests(str(tmp_path), boot_token="b1"))
    assert len(xm.load_manifests(str(tmp_path), boot_token="b1")) == (
        len(P_CUT) + D_RANKS)


def test_a_manifest_of_another_schema_version_refuses(tmp_path):
    raw = _manifests()[0].as_json()
    raw["version"] = xm.MANIFEST_VERSION + 1
    (tmp_path / xm.manifest_filename(0, "P")).write_text(json.dumps(raw))
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.load_manifests(str(tmp_path))
    assert "W68" in str(exc.value)


def test_the_write_site_reads_no_tensor_and_re_walks_nothing():
    """The write-along writes down what the loader decided; it does not
    reconstruct it.  A second walk here would reintroduce, in the same commit,
    exactly the drift this slice removes.

    OVER THE AST, NOT OVER THE TEXT, and that is a finding about this test
    rather than a nicety: the first version grepped the source and went red on
    its OWN MODULE DOCSTRING, which explains the defect by NAMING
    ``data_ptr()``.  That is the #995 prose-marker trap inside a unit test --
    a marker counted where it was merely mentioned -- and a rule that cannot
    tell code from prose would have forced the explanation out of the file.
    """
    import ast
    import inspect

    # SCOPED TO THE WRITE PATH, and the narrowing is a correction of THIS
    # test rather than a loosening of the rule.  The first version asserted it
    # over the whole module and then fired on the MATERIALISATION CHECK, which
    # reads a tensor ON PURPOSE -- at LEG time, comparing the manifest written
    # at the end of loading against the hardware now, which is the one moment
    # the comparison is not tautological.  The rule was always about the WRITE
    # path: what is recorded must be the loader's decision, not a second walk
    # of the same tensors.  Asserting it module-wide would have forced the
    # drift check out of the file, i.e. a true rule applied at the wrong scope
    # deleting a real guard.
    for fn in (xm.pieces_from_inventory, xm.write_this_rank,
               xm.write_rank_manifest, xm.manifest_filename):
        tree = ast.parse(inspect.getsource(fn))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("data_ptr", "named_parameters", "named_buffers",
                          "element_size", "cuda"):
            assert forbidden not in called, (
                f"{fn.__name__} CALLS {forbidden}(): the write path records "
                f"the loader's decision and never re-walks the tensors")

    tree = ast.parse(inspect.getsource(xm))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "torch" not in imported


def test_the_product_write_site_is_the_end_of_weight_loading():
    """``arm_coverage_at_load`` -- proven reachable on metal (44/44 COVER lines
    on weg2xsn17/18/19), and it may not raise (no group fence here)."""
    import inspect

    src = inspect.getsource(wx.arm_coverage_at_load)
    assert "_write_placement_manifest" in src
    helper = inspect.getsource(wx._write_placement_manifest)
    assert "except BaseException" in helper, (
        "a rank-local raise at the end of weight loading leaves the other five "
        "in a collective without six members (refuter F5)")


# ---------------------------------------------------------------------------
# (3) THE JOIN
# ---------------------------------------------------------------------------


def test_the_join_reads_the_shard_axis_off_the_two_groups_records():
    """Neither guessed nor configured: row cut, column cut, replica, refusal."""
    join = _join()
    by = join.by_name
    q = by["model.layers.0.self_attn.qkv_proj.weight"]
    assert q.shard_axis == wx.ROWS
    assert q.rows_full == 1536 and q.cols_full == 512
    assert q.tp_widths == _split(1536, D_RANKS)
    d = by["model.layers.0.mlp.down_proj.weight"]
    assert d.shard_axis == wx.COLS
    assert d.rows_full == 2048 and d.cols_full == 1024
    assert d.tp_widths == _split(1024, D_RANKS)
    # A replica is a replica and is NOT called a cut.
    assert _join(sharded=False).by_name[q.param_name].shard_axis == wx.REPLICATED


def test_the_unsharded_extent_is_the_joins_answer_no_rank_holds_it():
    """``derive_leg_plan``'s docstring (:3121): the unsharded extent is
    cross-group knowledge.  Here it is the destination's rows, summed."""
    join = _join()
    for t in join.tensors:
        if t.shard_axis == wx.ROWS:
            assert sum(t.tp_widths) == t.rows_full
        elif t.shard_axis == wx.COLS:
            assert sum(t.tp_widths) == t.cols_full


def test_the_source_of_every_tensor_is_the_p_stage_of_its_layer():
    """THE JOIN'S POINT: the destination learns the holder from a FILE the
    holder wrote, not from a pointer it cannot read."""
    join = _join()
    for t in join.tensors:
        layer = int(t.param_name.split(".")[2])
        assert t.pp_stage == _stage_of_layer(layer), t.param_name
        assert t.pp_card == CARDS[t.pp_stage]


def test_a_tensor_with_no_counterpart_refuses_by_name():
    mans = _manifests()
    victim = "model.layers.0.mlp.down_proj.weight"
    mans = [
        m if m.group != "P" else xm.RankManifest(
            group=m.group, rank=m.rank, card=m.card, region_tag=m.region_tag,
            boot_token=m.boot_token,
            pieces=tuple(p for p in m.pieces if p.param_name != victim))
        for m in mans
    ]
    with pytest.raises(wx.Weg2XchgSourceMissing) as exc:
        xm.join_manifests(mans, pp_group="P", tp_group="D")
    assert "W74" in str(exc.value) and victim in str(exc.value)


def test_a_shape_contradiction_refuses_and_never_degrades_to_replicated():
    """Falling back to REPLICATED is the tree's own silent answer
    (weight_exchange_shadow.py:3237) and the reason the cut was invisible."""
    mans = _manifests()
    victim = "model.layers.0.self_attn.qkv_proj.weight"
    out = []
    for m in mans:
        if m.group == "D" and m.rank == 0:
            pieces = tuple(
                _piece(p.param_name, p.rows_full + 3, p.cols_full, p.itemsize)
                if p.param_name == victim else p for p in m.pieces)
            m = xm.RankManifest(group=m.group, rank=m.rank, card=m.card,
                                region_tag=m.region_tag,
                                boot_token=m.boot_token, pieces=pieces)
        out.append(m)
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.join_manifests(out, pp_group="P", tp_group="D")
    assert "W68" in str(exc.value) and victim in str(exc.value)


def test_a_missing_group_refuses_rather_than_planning_over_who_published():
    with pytest.raises(wx.Weg2XchgSourceMissing) as exc:
        xm.join_manifests([m for m in _manifests() if m.group == "D"],
                          pp_group="P", tp_group="D")
    assert "W74" in str(exc.value)


def test_the_join_line_carries_every_denominator():
    line = _join().line(direction=wx.LEGS_PP_TO_TP)
    assert "pp=P tp=D" in line
    assert f"direction={wx.LEGS_PP_TO_TP}" in line
    assert f"pp_ranks={len(P_CUT)} tp_ranks={D_RANKS}" in line
    assert f"sharded={2 * N_LAYERS}/{2 * N_LAYERS}" in line
    assert "unsourced=0" in line


# ---------------------------------------------------------------------------
# (4) THE PROVIDER -- src_resolved=N/N ON A PLAN THAT ACTUALLY CUTS SHARDS
# ---------------------------------------------------------------------------


def test_a_diagonal_destination_layout_is_refused_by_name():
    """Operator ruling 2026-09-12: ``src_resolved=N/N`` on a diagonal plan does
    not count.  The tree's product path builds exactly that layout
    (weight_exchange_shadow.py:3332-3334, BOTH groups ``tp_size=1``), so
    without this refusal the slice's own acceptance number could be satisfied
    by the shape the slice exists to replace."""
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.refuse_diagonal_layout(1, tp_group="D")
    assert "W68" in str(exc.value) and "diagonal" in str(exc.value)
    xm.refuse_diagonal_layout(3, tp_group="D")      # a real TP group passes


def test_the_pointer_only_path_still_resolves_no_source():
    """THE RED ANCHOR, and it must STAY green: with no source address book the
    plan reads 0/N -- the measured XSN20 shape, as a property of that path."""
    join = _join()
    plan = xm.plan_from_join(join, src_addr=None,
                             dst_addr=lambda name, rank: 0x1000)
    prof = wx.pointer_profile(plan.descs)
    assert prof.descs_total > 0
    assert prof.src_resolved == 0
    assert prof.dst_resolved == prof.descs_total


def test_the_join_backed_provider_resolves_both_sides_and_cuts_shards():
    """THE GATE OF THIS SLICE.

    ``src_resolved=N/N`` AND a real shard cut: more descriptors than tensors
    on the sharded classes, because each destination rank takes its own slice
    of a tensor one P stage holds whole.  The ADDRESS is still the caller's
    (here the host counter-proof buffer); the IDENTITY -- which source covers
    which destination slice -- is the join's, which is what a peer's manifest
    can answer and a peer's ``data_ptr()`` never can.
    """
    join = _join()
    book = {}

    def src_addr(name, rank):
        return book.setdefault((name, rank), 0x7000_0000 + 4096 * len(book))

    plan = xm.plan_from_join(join, src_addr=src_addr,
                             dst_addr=lambda name, rank: 0x1000 + 8 * rank)
    prof = wx.pointer_profile(plan.raw_descs)
    assert prof.descs_total > 0
    assert prof.src_resolved == prof.descs_total, (
        f"src_resolved={prof.src_resolved}/{prof.descs_total}")
    assert prof.dst_resolved == prof.descs_total

    # THE SHARD CUT, which a diagonal plan cannot produce.
    assert len(plan.raw_descs) > len(join.tensors), (
        "a plan with no more descriptors than tensors moved whole tensors, "
        "i.e. it is the diagonal this slice replaces")
    per_name = {}
    for d in plan.raw_descs:
        per_name.setdefault(d.param_name, set()).add(d.dst_rank)
    sharded = [n for n, ranks in per_name.items() if len(ranks) == D_RANKS]
    assert sharded, "no tensor was cut across all three destination ranks"

    # And the source of every descriptor is the P stage of that layer.
    for d in plan.raw_descs:
        if d.kind == wx.ZEROFILL:
            continue
        assert d.src_rank == _stage_of_layer(int(d.param_name.split(".")[2]))


def test_the_bounce_finds_no_hole_in_a_join_backed_leg():
    """The end of the XSN20 wall at the exact function that raised it:
    ``weight_exchange_bounce._missing_pointer`` (:746)."""
    from sglang.srt.weg2 import weight_exchange_bounce as xb

    join = _join()
    good = xm.plan_from_join(join, src_addr=lambda n, r: 0x7000_0000,
                             dst_addr=lambda n, r: 0x1000)
    assert xb._missing_pointer(good.descs) is None
    blind = xm.plan_from_join(join, src_addr=None,
                              dst_addr=lambda n, r: 0x1000)
    assert "has no source pointer" in (xb._missing_pointer(blind.descs) or "")


def test_materialisation_drift_refuses_by_name():
    """Sharp from commit 1: a manifest that has drifted from the hardware is a
    WRONG answer, and every downstream reader trusts it."""

    class _T:
        def __init__(self, rows, cols, item):
            self.shape = (rows, cols)
            self._s = (cols, 1)
            self._i = item

        def stride(self):
            return self._s

        def element_size(self):
            return self._i

        def dim(self):
            return 2

    p = _piece("model.layers.0.mlp.down_proj.weight", 2048, 1024)
    xm.refuse_on_materialisation_drift(p, _T(2048, 1024, 1))   # silent
    for live in (_T(2049, 1024, 1), _T(2048, 1023, 1), _T(2048, 1024, 2)):
        with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
            xm.refuse_on_materialisation_drift(p, live)
        assert "W68" in str(exc.value) and p.param_name in str(exc.value)


def test_the_launcher_publishes_a_manifest_directory_the_ranks_can_actually_read():
    """THE WRITER'S OWN 'BUILT BUT NEVER EXECUTED' TRAP, pinned.

    The manifest was first keyed on ``SGLANG_PHASE_FOOTPRINT_DUMP``.  That
    variable reaches a weg2 rank ONLY under ``--xchg-coverage-diff``
    (``launcher.coverage_dump_dir`` returns ``""`` otherwise, deliberately, so
    the instrument's OFF state is byte-identical), so the writer would have
    found an empty directory on every ordinary boot, returned ``None``, and
    left no file and no line.  Desk-green, metal-silent.

    This asserts the publisher: the directory rides the xchg env, which is
    published whenever the arm is armed, and is POPPED on the ring arm.
    """
    import inspect

    from sglang.srt.weg2 import launcher

    pub = inspect.getsource(launcher.prepare_xchg_env)
    assert "xchg_manifest.DIR_ENV" in pub
    assert "manifest_dir" in inspect.signature(
        launcher.prepare_xchg_env).parameters
    assert "xchg_manifest.DIR_ENV" in inspect.getsource(launcher.build_env)

    env = launcher.prepare_xchg_env(lambda *a, **k: None, "epoch1", "shadow",
                                    dry=True, manifest_dir="/tmp/evi")
    assert env.get(xm.DIR_ENV) == "/tmp/evi"
    assert launcher.prepare_xchg_env(
        lambda *a, **k: None, "epoch1", "ring", dry=True) == {}, (
        "the ring arm must publish nothing at all")


# ---------------------------------------------------------------------------
# (5) BOTH DIRECTIONS, OUT OF THE ONE JOIN (user correction 2026-09-12:
#     "PP->TP und TP->PP gleichwertig und gleichzeitig")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP])
def test_both_directions_resolve_both_sides_and_cut_the_same_shards(direction):
    """ONE join, two directions, equal citizens.

    The mirror direction is where a role-oriented join fails silently, and the
    failure has a name: ``_blocks_of`` honours a per-tensor width vector ONLY
    on the destination (``weight_exchange.py:1575``); a SOURCE at ``tp_size>1``
    goes through ``layout.ratios_for(geom.family)`` (``:1594``).  Under
    ``tp_to_pp`` the TP group IS the source, so a plan that only filled
    ``dst_widths`` would fall back to an EVEN split on exactly the side that is
    unevenly cut -- on both ends equally, with nothing downstream able to see
    it (the #1275 class).  Keying ``family_ratios`` by the parameter name is
    what closes it; MUTANT: drop ``family=`` from ``JoinedTensor.geom`` and the
    tp_to_pp arm of this test dies on the shard boundary.
    """
    join = _join()
    book = {}

    def addr(name, rank):
        return book.setdefault((name, rank, "s"), 0x7000_0000 + 4096 * len(book))

    plan = xm.plan_from_join(join, direction=direction, src_addr=addr,
                             dst_addr=lambda n, r: 0x1000 + 64 * r)
    prof = wx.pointer_profile(plan.raw_descs)
    assert prof.src_resolved == prof.descs_total > 0, (
        f"{direction}: src_resolved={prof.src_resolved}/{prof.descs_total}")
    assert prof.dst_resolved == prof.descs_total
    assert len(plan.raw_descs) > len(join.tensors), (
        f"{direction}: no more descriptors than tensors -- whole tensors moved")

    tp_side = "dst_rank" if direction == wx.LEGS_PP_TO_TP else "src_rank"
    pp_side = "src_rank" if direction == wx.LEGS_PP_TO_TP else "dst_rank"
    per_name = {}
    for d in plan.raw_descs:
        if d.kind == wx.ZEROFILL:
            continue
        per_name.setdefault(d.param_name, set()).add(getattr(d, tp_side))
        # The PP side is always the ONE stage that holds the layer.
        assert getattr(d, pp_side) == _stage_of_layer(
            int(d.param_name.split(".")[2])), (direction, d.param_name)
    assert per_name, direction
    for name, ranks in per_name.items():
        assert ranks == {0, 1, 2}, (direction, name, ranks)


def test_the_uneven_cut_survives_the_mirror_direction_exactly():
    """The boundaries must be IDENTICAL in both directions, not merely present.

    An even split would also produce three ranks and the right byte total --
    which is the danger direction the briefing names: a wrong slice with the
    right byte numbers.  This compares the actual unit ranges.
    """
    join = _join()
    ranges = {}
    for direction in (wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP):
        plan = xm.plan_from_join(join, direction=direction,
                                 src_addr=lambda n, r: 0x7000_0000,
                                 dst_addr=lambda n, r: 0x1000)
        tp_side = "dst_rank" if direction == wx.LEGS_PP_TO_TP else "src_rank"
        seen = {}
        for d in plan.raw_descs:
            if d.kind == wx.ZEROFILL:
                continue
            seen.setdefault(d.param_name, {})[getattr(d, tp_side)] = int(
                d.rows if d.spitch else d.nbytes)
        ranges[direction] = seen
    assert ranges[wx.LEGS_PP_TO_TP].keys() == ranges[wx.LEGS_TP_TO_PP].keys()

    # And the widths are the JOIN's, not an even split.
    name = "model.layers.0.self_attn.qkv_proj.weight"
    widths = join.by_name[name].tp_widths
    assert widths == _split(1536, D_RANKS)
    assert len(set(widths)) > 1, "the fixture must be UNEVEN or this proves nothing"
    assert widths != (1536 // 3,) * 3


def test_an_unknown_direction_refuses_rather_than_defaulting():
    join = _join()
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.plan_from_join(join, direction="pp2tp",
                          src_addr=lambda n, r: 1, dst_addr=lambda n, r: 2)
    assert "W68" in str(exc.value)


def test_the_join_is_group_oriented_and_names_no_role():
    """A role-keyed join is two derivations waiting to happen."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(xm.JoinedTensor)}
    assert "pp_stage" in fields and "tp_widths" in fields
    assert not {"src_rank", "dst_rank", "src_widths", "dst_widths"} & fields, (
        "the join must not name a role: the direction chooses roles, the join "
        "only holds extents")


def test_the_module_never_reads_the_source_bytes_out_of_the_ring():
    """THE RING IS THE COUNTER-PROOF, NEVER THE SOURCE (operator, 2026-09-12).

    Resolving the source side out of the ring would be the ring restore
    through another door, and it defeats the exchange's whole goal: zero layer
    bytes resident in host RAM. This module must therefore know nothing about
    the ring or any carrier -- the address books are the caller's.
    """
    import inspect

    src = inspect.getsource(xm.plan_from_join)
    assert "src_addr" in src and "dst_addr" in src
    module = inspect.getsource(xm)
    for forbidden in ("tms_backup", "tms-backup", "host_ring", "TMS_HOST_RING",
                      "carrier_census", "ring_table"):
        assert forbidden not in module, (
            f"{forbidden!r} in xchg_manifest: the ring is the counter-proof, "
            f"never the source of the exchange")


# ---------------------------------------------------------------------------
# (6) TWO RUNNERS PER RANK -- the main model AND the drafter (AMENDMENT 6)
# ---------------------------------------------------------------------------


def _draft_manifest(group, rank):
    """What the DRAFT runner publishes from the same rank, same process."""
    return xm.RankManifest(
        group=group, rank=rank, card=CARDS[rank], region_tag="weights_draft",
        boot_token="b1",
        pieces=(_piece("model.layers.0.mtp.fc.weight", 64, 32,
                       tag="weights_draft"),))


def test_the_draft_runners_manifest_does_not_clobber_the_main_models():
    """PROVEN ON METAL BEFORE IT COULD BITE (boot weg2xsn20, D log).

    ``arm_coverage_at_load`` is called from ``ModelRunner.load_model``
    (``model_runner.py:2564``) with that runner's own ``weights_tag``
    (``:2461``), and a weg2 rank runs TWO runners in ONE process.  XSN20's D
    log carries three ``WEG2-XCHG-RESIDENT tag=weights_draft ... rank={0,1,2}``
    lines beside the ``weights_0`` ones, so the site fires twice per rank.

    Keyed on (group, rank) alone, the second write CLOBBERS the first and the
    join plans over whichever runner finished last -- from a file that looks
    complete.  AMENDMENT 6 makes that a loss of real exchanged bytes: the draft
    tag is IN the weights family.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        main = _manifests()[3]                      # D rank 0, weights_0
        draft = _draft_manifest("D", 0)
        a = xm.write_rank_manifest(main, tmp)
        b = xm.write_rank_manifest(draft, tmp)
        assert a != b, "the two runners of one rank must not share a file name"
        assert len(os.listdir(tmp)) == 2
        back = xm.load_manifests(tmp, boot_token="b1")
        assert len(back) == 2
        merged = xm.merge_region_tags(back)
        assert len(merged) == 1, "the join's unit is the RANK, not the runner"
        names = {p.param_name for p in merged[0].pieces}
        assert "model.layers.0.mtp.fc.weight" in names, "the draft bytes were lost"
        assert len(names) == len(main.pieces) + 1
        assert merged[0].region_tag == "weights_0+weights_draft"


def test_two_runners_disagreeing_about_one_tensor_refuse():
    """Picking either would be the rank-local derivation the manifest removes."""
    a = xm.RankManifest(group="D", rank=0, card=0, region_tag="weights_0",
                        boot_token="b1",
                        pieces=(_piece("shared.weight", 8, 4),))
    b = xm.RankManifest(group="D", rank=0, card=0, region_tag="weights_draft",
                        boot_token="b1",
                        pieces=(_piece("shared.weight", 9, 4),))
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.merge_region_tags([a, b])
    assert "W68" in str(exc.value) and "shared.weight" in str(exc.value)


def test_the_join_still_works_when_every_rank_has_two_runner_files():
    """End to end: six main files + six draft files -> one join, no refusal."""
    mans = list(_manifests())
    for group in ("P", "D"):
        for rank in range(3):
            mans.append(xm.RankManifest(
                group=group, rank=rank, card=CARDS[rank],
                region_tag="weights_draft", boot_token="b1",
                pieces=(_piece("model.layers.0.mtp.fc.weight",
                               *( (64, 32) if group == "P"
                                  else (_split(64, D_RANKS)[rank], 32) ),
                               tag="weights_draft"),)))
    join = xm.join_manifests(mans, pp_group="P", tp_group="D")
    assert "model.layers.0.mtp.fc.weight" in join.by_name
    draft = join.by_name["model.layers.0.mtp.fc.weight"]
    assert draft.shard_axis == wx.ROWS
    assert draft.tp_widths == _split(64, D_RANKS)
    assert join.unsourced == ()


# ---------------------------------------------------------------------------
# (7) THE RATCHET: the diagonal may never step in silently again
# ---------------------------------------------------------------------------


def test_no_join_path_can_reach_derive_leg_plan():
    """THE NO-FALLBACK RATCHET (operator order 2026-09-12).

    ``_weg2_shadow_plan`` branches on ``exchange_armed()``: armed -> the
    manifest join, every other arm -> the rank-local derivation.  What must be
    impossible is a route from the ARMED branch BACK to the derivation, because
    that derivation builds the on-card diagonal (``shard_axis=REPLICATED``,
    both GroupLayouts ``tp_size=1``) and asks a rank for the peer's pointer --
    measured as ``src_resolved=0/N`` on all 24 legs of weg2xsn20.

    Read over the AST rather than the text: the method's own comments NAME
    ``derive_leg_plan`` in order to explain why it must not be reached, and a
    grep cannot tell the explanation from the defect (third instance of that
    trap in this slice).
    """
    import ast
    import inspect
    import textwrap

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = textwrap.dedent(inspect.getsource(
        wu.SchedulerWeightUpdaterManager._weg2_shadow_plan))
    tree = ast.parse(src)

    # Find the `if <...exchange_armed()...>:` branch and prove that no call to
    # derive_leg_plan lives inside it, and that it always leaves the method.
    armed_branches = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "exchange_armed" in ast.dump(node.test)
    ]
    assert len(armed_branches) == 1, (
        "expected exactly one exchange_armed() branch in the product provider")
    branch = armed_branches[0]
    called = {
        n.func.attr for n in ast.walk(branch)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "derive_leg_plan" not in called, (
        "the armed branch can reach the rank-local derivation -- that is the "
        "silent fallback to the diagonal this ratchet exists to forbid")
    assert "leg_plan_from_join" in called and "manifests_for_boot" in called

    # Every exit of the armed branch is a return: nothing may fall through it
    # into the derivation below.
    tails = [n for n in branch.body if isinstance(n, (ast.Return, ast.If))]
    assert tails, "the armed branch must return, never fall through"
    assert isinstance(branch.body[-1], ast.Return), (
        "the armed branch's last statement must be a return; falling through "
        "would reach derive_leg_plan with the arm armed")
    assert not branch.orelse, (
        "an else here would make the two producers look like alternatives; "
        "the derivation is the UNARMED path and sits after the branch")


def test_a_missing_peer_manifest_names_the_expected_file(tmp_path, monkeypatch):
    """The refusal must be actionable: the PATH, not merely the condition."""
    monkeypatch.setenv(xm.DIR_ENV, str(tmp_path))
    monkeypatch.setenv("SGLANG_WEG2_XCHG_BOOT", "tok")
    for man in _manifests():
        if man.group == "P" and man.rank == 2:
            continue
        xm.write_rank_manifest(
            xm.RankManifest(group=man.group, rank=man.rank, card=man.card,
                            region_tag=man.region_tag, boot_token="tok",
                            pieces=man.pieces), str(tmp_path))
    mans, why = xm.manifests_for_boot()
    assert mans is None
    assert "manifest-missing" in why
    assert "phase_manifest_P_rank2_" in why and ".json" in why
    assert "diagonal" in why, (
        "the refusal must say WHY there is no fallback, or the next reader "
        "adds one back")
