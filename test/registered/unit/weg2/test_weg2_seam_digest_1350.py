# SPDX-License-Identifier: Apache-2.0
"""#1350: THE SEAM GRADER -- did the exchange bring MY bytes back?

WHY THIS FILE EXISTS (operator ruling 2026-09-11 ~16:2xZ, PLAN_S6_BOUNCE_0911.md
"RE-STAMP-11-PRAEMISSE WIDERLEGT, GRADER WIRD GEBAUT (Variante a)").

RE-STAMP 11 moved the byte comparison onto the authoritative path expecting to
find a SECOND copy of the weight bytes there.  There is none, and the reason is
a user law rather than a defect: TMS frees the pinned host image right after the
H2D restore (``cudaFreeHost(metadata.cpu_backup_granules[k])`` plus ``clear()``
at both sites in ``tms_csrc/core.cpp``), which is
``gewichtsaustausch-ziel-kein-dauer-hostram`` implemented literally.  A
byte-faithful ``memcmp`` needs two rank-local copies; the law guarantees exactly
one.  So the memcmp is as unfoundable on the authoritative seam as on the shadow
seam, from a different and STRONGER reason.

Variant (b) -- "no byte comparison is foundable, close the item" -- would have
been the honest FINDING and an insufficient RESOLUTION: it ships the exchange
without a grader, which is the class this campaign has paid for all day
(green-by-vacancy, counter-vs-actuator, delivered-never-wired).

So variant (a): A RANK-LOCAL DIGEST OVER THE SEAM.  Taken before the pause,
recomputed after the landing, same rank, same pieces.  Both sides rank-own: no
peer, no collective, no byte movement between cards, both moments OUTSIDE the
no-return region (#875 DO-NOT-BUILD untouched).  Cost is a hash, not an image,
so the host-RAM law stays intact.  It is even SUPERIOR to the memcmp it
replaces: a memcmp against a second copy would have proven COPY fidelity; this
proves ROUND-TRIP fidelity, which is the question the exchange must answer.

THE KEY IS THE PLACEMENT (operator direction 2026-09-12).  The grader does NOT
walk ``model.named_parameters()`` itself and does not invent a
``(name, shape, dtype)`` triple: placement is a pure function of (header, P cut,
D vector, quantisation), decided by the LOADER at boot, and the tree already
carries it as ``ParamGeom`` / ``XchgDesc`` through ``build_plan`` and
``derive_card_manifest``.  A second inventory would be second bookkeeping -- the
W80/W84/W19 family.  So the piece enumeration comes from ONE producer,
``weight_exchange_shadow.card_inventory``, and the key is
``(manifest_entry(name, class, rows, cols, itemsize), tag, card)``.  That is
what buys the localisation: a mismatch NAMES the pieces.

THE THREE OPERATOR CONDITIONS ARE DESIGN CONDITIONS, NOT ADDENDA, and each has
its own test below plus a mutant that must die:

  (1) LOGICAL CONTENT PER PIECE, never an arena span, keyed by PLACEMENT.  The
      exchange moves the arena BY DESIGN, so a span digest would compare two
      different things and be Instrument-lies class A.
      -> `test_the_digest_survives_an_arena_change_that_preserves_content`
      -> `test_the_byte_count_is_the_logical_extent_not_the_storage_span`
      -> `test_the_key_carries_no_pointer_no_pitch_no_arena`
      -> mutant M1 (span instead of content) dies on those.
  (2) DEFAULT OFF, armed only for instrument boots (S6I shape) through a NAMED
      knob.  Hashing a whole shard per rank is not free; always-on would be a
      flip cost regression, i.e. a performance defect the grader causes.
      -> `test_default_is_off_and_not_one_byte_is_read`
      -> mutant M2 (always-on) dies there.
  (3) AN HONEST LIMIT in the log line and in the record.  AMENDED BY THE
      RE-KEYING, and the amendment is stated rather than assumed: the operator's
      original wording was "a digest DETECTS, it does not LOCALIZE".  With the
      placement key that hole is closed one level -- a mismatch names the pieces
      -- and what remains true is that it does not localise WITHIN a piece and
      that a green digest is never "layout verified".
      -> `test_the_verdict_line_carries_the_honest_limit_verbatim`
      -> `test_a_content_mismatch_names_the_pieces_that_moved`

AWAKE-ONLY IS BINDING, and it is a STRUCTURAL fact rather than a cost choice
(`understand_tensor-map.md` section 5.4 + ``weg2_memory_saver.py:123``,
``WEG2_SLEEP_TAGS = {kv_cache, weights}``): at flip time the sleeping group
holds NO weight bytes in VRAM at all.  A "compare at the cutover" design is not
expensive, it is IMPOSSIBLE.
  -> `test_the_only_call_sites_are_the_two_awake_seams`
  -> `test_no_cutover_or_flip_module_takes_a_reading`
  -> mutant M4 (a reading at the cutover) dies there.

THE DANGER DIRECTION NOBODY WOULD SEE: a comparison ACROSS TWO PLACEMENTS.  If
the piece set moved, the content digests describe different things and comparing
them is not a finding.  The compare REFUSES by naming the placement.
  -> `test_a_changed_placement_refuses_the_content_compare_by_name`
  -> mutant M3 (placement key ignored) dies there.

AND THE ONE THIS PROJECT KEEPS PAYING FOR: a verdict that is produced, logged
and then not acted upon.  A MISMATCH RAISES.
  -> `test_a_mismatch_is_a_named_refusal_not_a_log_line`
  -> `test_the_wake_hook_raises_the_named_refusal`
  -> mutant M5 (log, do not refuse) dies on both.

RED-FIRST STATE AT ``d294bde41d``: ``sglang.srt.weg2.seam_digest`` does not
exist and ``weight_exchange_shadow.card_inventory`` does not exist, so every
test below fails by NAME (the imports are deferred into `_mod()` / `_shadow()`
for exactly that reason -- a module-level import would collapse the whole file
into one collection error and prove nothing per name).
"""

import ast
import importlib
import inspect
import os
import re
import textwrap

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _mod():
    """The module under test, imported LATE so each test reds by its own name."""
    return importlib.import_module("sglang.srt.weg2.seam_digest")


def _shadow():
    return importlib.import_module("sglang.srt.weg2.weight_exchange_shadow")


def _wu():
    from sglang.srt.managers.scheduler_components import weight_updater

    return weight_updater.SchedulerWeightUpdaterManager


# ---------------------------------------------------------------------------
# Fixtures.  The inventory is (ParamGeom, tensor) pairs built with EXACTLY the
# arguments `card_inventory` builds them with -- REPLICATED, shard_total 0,
# stage = this rank -- so the test drives the product's own geometry rather
# than a look-alike.
# ---------------------------------------------------------------------------
_TENSORS = (
    ("model.layers.0.self_attn.qkv_proj.weight", "weights_0", (64, 40), torch.bfloat16),
    ("model.layers.0.mlp.down_proj.weight", "weights_0", (40, 96), torch.float32),
    ("model.layers.0.input_layernorm.weight", "weights_0", (40,), torch.float32),
    ("model.embed_tokens.weight", "weights", (128, 40), torch.bfloat16),
)


def _tensor(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(dtype)


def _inventory(seed: int = 7, rank: int = 1):
    out = []
    for i, (name, tag, shape, dtype) in enumerate(_TENSORS):
        t = _tensor(shape, dtype, seed + i)
        geom = wx.ParamGeom.of(
            t, name=name, tag=tag, shard_axis=wx.REPLICATED, shard_total=0,
            stage=int(rank),
        )
        out.append((geom, t))
    return out


def _rearena(inventory, rank: int = 1):
    """The SAME logical values, in a different ARENA -- the change the exchange
    actually makes.

    Every property that moves when the bytes are re-laid is moved here and the
    content is not:

    * a FRESH allocation (other ``data_ptr``, other storage object),
    * a non-zero ``storage_offset``,
    * a PADDED PITCH, so the storage extent is strictly larger than the piece,
    * and the pad is filled with GARBAGE rather than zeros, so a digest over the
      storage SPAN genuinely differs while the content digest does not.

    The placement identity is unchanged by construction: ``manifest_entry``
    carries ``(name, class, rows, cols, itemsize)`` and deliberately not the
    pitch, for the same reason this test exists.
    """
    out = []
    for geom, t in inventory:
        if t.dim() == 1:
            big = torch.full((t.shape[0] + 11,), 7.0, dtype=t.dtype)
            view = big[5 : 5 + t.shape[0]]
        else:
            big = torch.full((t.shape[0], t.shape[1] + 7), 7.0, dtype=t.dtype)
            view = big[:, : t.shape[1]]
        view.copy_(t)
        new_geom = wx.ParamGeom.of(
            view, name=geom.name, tag=geom.tag, shard_axis=wx.REPLICATED,
            shard_total=0, stage=int(rank),
        )
        out.append((new_geom, view))
    return out


def _reading(mod, stage, inventory, *, tags=("weights",), rank=1, card=2, epoch=4):
    return mod.take_reading(
        stage, inventory, group="P", rank=rank, card=card, tags=tags, epoch=epoch
    )


# ===========================================================================
# THE KEY: the tree's published placement identity, not a second inventory.
# ===========================================================================


def _code_identifiers(module):
    """Every name the module's CODE uses -- docstrings and comments excluded.

    ON THE AST AND NOT A TEXT SCAN, and the reason is measured rather than
    stylistic.  The first version of this test grepped the source and went red
    on the module's OWN DOCSTRING, which says in prose that it no longer walks
    ``model.named_parameters()``.  That is the #995 prose-marker trap one level
    up -- an explanation of what was removed read as the thing itself -- and it
    is the same reason #1273 B4f moved its wiring pins onto ``main()``'s AST
    after a text scan let a mutant through.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(module)))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out


def test_the_grader_reads_the_producer_and_builds_no_second_inventory():
    """The operator's direction of 2026-09-12, asserted on the module's CODE.

    A private ``named_parameters()`` walk here would be a second reading of a
    fact the loader already decided and ``card_inventory`` already publishes --
    the W80/W84/W19 family.  The producer is named; a copy of it is not.
    """
    used = _code_identifiers(_mod())
    assert "named_parameters" not in used, (
        "the grader walks the live model itself: that is the second inventory "
        "the placement re-keying removed"
    )
    # The grader's own identity comes from the published producers, never from
    # a private re-implementation of them.
    assert "manifest_entry" in used and "tensor_class" in used

    # AND THE INVENTORY COMES FROM THE PRODUCER AT THE HOOK, not here.  The
    # module is pure -- it is HANDED (ParamGeom, tensor) pairs -- so asserting
    # `card_inventory` against THIS file would be asserting against the wrong
    # one; that mis-scoping is what the first run of this test caught in
    # itself.  The obligation is real, it just belongs to the caller.
    hook = _code_identifiers(_wu()._weg2_seam_inventory)
    assert "card_inventory" in hook, (
        "the seam hook does not read the shared producer"
    )
    assert "named_parameters" not in hook


def test_card_inventory_is_the_one_producer_the_manifest_also_uses():
    """ONE producer, two consumers -- so the manifest the co-located pair agrees
    over and the pieces this grader hashes cannot be two different sets."""
    shadow = _shadow()
    assert callable(shadow.card_inventory)
    src = inspect.getsource(shadow.derive_card_manifest)
    assert "card_inventory(" in src, (
        "derive_card_manifest no longer reads the shared producer: the walk has "
        "been copied instead of extracted"
    )


def test_the_key_carries_no_pointer_no_pitch_no_arena():
    """Condition (1) at the level of the KEY itself.

    The four properties that move when the arena moves must not be in it.  A
    key that carried any of them would fire on every correct flip.
    """
    mod = _mod()
    inv = _inventory()
    re_inv = _rearena(inv)
    for (geom, t), (re_geom, re_t) in zip(inv, re_inv):
        a = mod.identity_of(geom, card=2)
        b = mod.identity_of(re_geom, card=2)
        assert a.key == b.key, f"{a.label}: the key moved with the arena"
        assert t.data_ptr() != re_t.data_ptr()
        assert t.untyped_storage().nbytes() != re_t.untyped_storage().nbytes()
    # and the key DOES move with the coordinates that are placement
    geom = inv[0][0]
    assert mod.identity_of(geom, card=2).key != mod.identity_of(geom, card=3).key
    other_tag = geom.replace(tag="weights_5")
    assert mod.identity_of(other_tag, card=2).key != mod.identity_of(geom, card=2).key


def test_the_identity_label_names_the_piece_a_human_would_look_for():
    mod = _mod()
    identity = mod.identity_of(_inventory()[0][0], card=2)
    assert identity.param_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert identity.cls == "qkv_proj"
    assert identity.tag == "weights_0"
    assert identity.card == 2
    for part in ("qkv_proj", "weights_0", "card2"):
        assert part in identity.label


# ===========================================================================
# CONDITION (1): content, not span.
# ===========================================================================


def test_the_digest_survives_an_arena_change_that_preserves_content():
    """THE HARD HALF OF THE RED-FIRST FALSIFIER, and condition (1) itself.

    The exchange re-lays the bytes by design.  A digest that moved when only
    the arena moved would report MISMATCH on every correct flip -- the
    instrument would be the defect.  Mutant M1 (hash the storage span) dies
    here, and it dies loudly because the pad bytes are garbage, not zeros.
    """
    mod = _mod()
    before = _reading(mod, "before", _inventory(), tags=("weights_0", "weights"))
    after = _reading(mod, "after", _rearena(_inventory()), tags=("weights",))
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MATCH, (
        f"an arena change that preserves content was graded "
        f"{verdict.verdict}/{verdict.reason}: this is a span digest, not a "
        f"content digest. moved={verdict.moved}"
    )
    assert before.digest == after.digest
    assert before.placement_key == after.placement_key


def test_the_byte_count_is_the_logical_extent_not_the_storage_span():
    """Second face of M1: even the reported ``bytes`` must be the logical one.

    The re-arena'd pieces sit in padded allocations.  A reading that priced the
    span would print a number no reader could reconcile with the plan, and
    would be the same category error one level down.
    """
    mod = _mod()
    inv = _inventory()
    logical = sum(int(t.numel()) * int(t.element_size()) for _g, t in inv)
    assert _reading(mod, "before", inv).bytes == logical
    assert _reading(mod, "after", _rearena(inv)).bytes == logical


def test_a_single_flipped_element_after_the_landing_is_a_mismatch():
    """THE OTHER HALF OF THE FALSIFIER: the direction the grader exists for."""
    mod = _mod()
    before = _reading(mod, "before", _inventory())
    landed = _rearena(_inventory())
    landed[1][1][3, 7] = landed[1][1][3, 7] + 1.0
    after = _reading(mod, "after", landed)
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MISMATCH
    assert verdict.reason == mod.REASON_CONTENT


def test_a_content_mismatch_names_the_pieces_that_moved():
    """CONDITION (3) AS AMENDED: the localisation the placement key buys.

    This is the half the pre-re-keying design could not do -- it could say THAT
    something moved and not WHICH.  A count alone is the hole; the labels close
    it.
    """
    mod = _mod()
    before = _reading(mod, "before", _inventory())
    landed = _rearena(_inventory())
    landed[1][1][0, 0] = landed[1][1][0, 0] + 1.0
    verdict = mod.compare(before, _reading(mod, "after", landed))
    assert verdict.moved == (
        "model.layers.0.mlp.down_proj.weight[down_proj]@weights_0/card2",
    )
    line = verdict.line()
    assert "moved=1" in line
    assert "down_proj" in line
    # the pieces that did NOT move are not named
    assert "qkv_proj" not in line


def test_the_digest_is_not_a_count_or_a_placement_check():
    """DENOMINATOR FIRST: a digest that hashed only the placement would pass the
    two tests above and detect nothing.  Same placement, different content."""
    mod = _mod()
    a = _reading(mod, "before", _inventory(seed=7))
    b = _reading(mod, "after", _inventory(seed=80))
    assert a.placement_key == b.placement_key, "the two readings must share a placement"
    assert a.digest != b.digest
    assert mod.compare(a, b).verdict == mod.VERDICT_MISMATCH


# ===========================================================================
# A comparison across two placements is refused, never graded.
# ===========================================================================


def test_a_changed_placement_refuses_the_content_compare_by_name():
    """Mutant M3 (ignore the placement key) dies here.

    Note what is asserted: not merely that the verdict is not MATCH -- a
    content compare across two placements also fails to match, so that
    assertion would survive M3 -- but that the REASON names the placement, and
    that the two sides are named.  Grading content across two placements is not
    a weaker finding, it is a different one.
    """
    mod = _mod()
    before = _reading(mod, "before", _inventory())
    moved = _inventory()[:-1]  # a piece is no longer on this card
    after = _reading(mod, "after", moved)
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MISMATCH
    assert verdict.reason == mod.REASON_PLACEMENT, (
        "a placement change was graded as a content change: the compare read "
        "content across two placements"
    )
    assert verdict.gone == ("model.embed_tokens.weight[embed_tokens]@weights/card2",)
    assert verdict.arrived == ()
    assert before.placement_key != after.placement_key
    line = verdict.line()
    assert f"placement_key_before={before.placement_key}" in line
    assert f"placement_key_after={after.placement_key}" in line
    assert "embed_tokens" in line


def test_a_piece_that_lands_on_another_card_is_a_placement_refusal():
    """The card is a placement coordinate, so a piece that comes back on the
    wrong card is refused by name rather than compared."""
    mod = _mod()
    before = _reading(mod, "before", _inventory(), card=2)
    after = _reading(mod, "after", _inventory(), card=3)
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MISMATCH
    assert verdict.reason == mod.REASON_PLACEMENT
    assert len(verdict.gone) == len(_TENSORS)
    assert len(verdict.arrived) == len(_TENSORS)


# ===========================================================================
# CONDITION (2): default OFF, armed by a NAMED knob through the #901 authority.
# ===========================================================================


class _Exploding(list):
    """An iterable that refuses to be read.  A whole shard per rank is the cost
    this guards; a test that only checked a boolean could not tell an unarmed
    hook from an armed hook whose result was thrown away."""

    def __iter__(self):
        raise AssertionError(
            "the unarmed path read the inventory: the digest is not default-off"
        )


def test_default_is_off_and_not_one_byte_is_read():
    """Mutant M2 (always-on) dies here."""
    mod = _mod()
    assert mod.seam_digest_armed({}) is False
    assert mod.take_reading_if_armed(
        "before", _Exploding(), group="P", rank=0, card=0, tags=(), epoch=1,
        environ={},
    ) is None


def test_the_arm_is_a_named_knob_resolved_by_the_901_authority():
    """No env rung without a flag, and the rung is reported, not assumed."""
    mod = _mod()
    from sglang.srt import knob_resolution as kr

    assert mod.ARM_FLAG == "--weg2-seam-digest"
    assert mod.ENV_ARM == "SGLANG_WEG2_SEAM_DIGEST"
    off = mod.arming_resolution({})
    assert isinstance(off, kr.Resolution)
    assert off.value is False
    assert off.source == kr.PROVENANCE_DEFAULT
    on = mod.arming_resolution({mod.ENV_ARM: "1"})
    assert on.value is True
    assert on.source == kr.env_source(mod.ENV_ARM)
    assert mod.seam_digest_armed({mod.ENV_ARM: "1"}) is True
    # A typo may never arm: the one direction an environment read may not take.
    assert mod.seam_digest_armed({mod.ENV_ARM: "yes-please"}) is False
    assert mod.seam_digest_armed({mod.ENV_ARM: ""}) is False
    # And the arming line names the flag an operator would have to pass.
    assert mod.ARM_FLAG in mod.arming_line({mod.ENV_ARM: "1"})


def test_the_launcher_publishes_the_arm_only_when_the_flag_is_set():
    """Launcher OUTPUT, popped when unarmed -- the same discipline every other
    weg2 env of this family follows, so an operator's inherited shell value can
    never arm a grader this boot did not ask for."""
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    assert "--weg2-seam-digest" in src
    # POPPED, not merely unset: the pop list is what makes an inherited value
    # harmless, and it is the half that is easy to forget.
    #
    # ON THE AST (#1273 B4f's lesson): a text scan for the name would also
    # match the comment that explains the pop, so it could pass on a tree where
    # the entry had been removed and only its justification left standing.
    tree = ast.parse(src)
    popped = set()
    published = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            if "'pop'" not in body:
                continue
            for element in node.iter.elts:
                if isinstance(element, ast.Attribute):
                    popped.add(f"{getattr(element.value, 'id', '?')}.{element.attr}")
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Attribute)
                and node.targets[0].slice.attr == "ENV_ARM"):
            published = True
    assert "seam_digest.ENV_ARM" in popped, (
        f"the grader's env is not in the launcher's pop list: {sorted(popped)}"
    )
    assert published, "the launcher never publishes the grader's env"


# ===========================================================================
# CONDITION (3): the honest limit, in the line and in the record.
# ===========================================================================


def test_the_verdict_line_carries_the_honest_limit_verbatim():
    mod = _mod()
    before = _reading(mod, "before", _inventory())
    after = _reading(mod, "after", _rearena(_inventory()))
    line = mod.compare(before, after).line()
    assert mod.LIMIT_CLAUSE in line
    assert "DETECTS" in mod.LIMIT_CLAUSE
    assert "does NOT localise within a piece" in mod.LIMIT_CLAUSE
    assert "layout verified" in mod.LIMIT_CLAUSE


def test_the_verdict_line_carries_every_field_the_boot_seat_reads():
    mod = _mod()
    before = _reading(mod, "before", _inventory(), tags=("weights_0",), rank=2, epoch=9)
    after = _reading(
        mod, "after", _rearena(_inventory()), tags=("weights",), rank=2, epoch=10
    )
    line = mod.compare(before, after).line()
    assert line.startswith(mod.LINE_PREFIX)
    for field in (
        "rank=2",
        "group=P",
        "card=2",
        "tag=",
        "placement_key_before=",
        "placement_key_after=",
        f"n_tensors={len(_TENSORS)}",
        f"bytes={before.bytes}",
        f"digest_before={before.digest}",
        f"digest_after={after.digest}",
        f"verdict={mod.VERDICT_MATCH}",
        "reason=",
        "moved=0",
        "population=",
    ):
        assert field in line, f"the verdict line has no {field!r}: {line}"


def test_a_long_mismatch_list_is_bounded_but_never_silently_truncated():
    mod = _mod()
    labels = tuple(f"p{i}[cls]@weights/card0" for i in range(mod.MAX_NAMED_PIECES + 5))
    v = mod.SeamVerdict(
        mod.VERDICT_MISMATCH, mod.REASON_CONTENT, None, None, moved=labels
    )
    line = v.line()
    assert f"moved={len(labels)}" in line
    assert "+5-more" in line


def test_unarmed_is_a_named_absence_and_never_reads_as_a_pass():
    """UNARMED is what a grader that produced NO verdict says.  It must be
    distinguishable from MATCH by anything that reads the line, and it must
    never raise either -- an absence is not a finding in either direction."""
    mod = _mod()
    v = mod.unarmed_verdict(
        "no-before-reading", group="P", rank=0, card=1, tags=("weights",)
    )
    assert v.verdict == mod.VERDICT_UNARMED
    assert v.verdict != mod.VERDICT_MATCH
    assert v.refusal() is None
    line = v.line()
    assert f"verdict={mod.VERDICT_UNARMED}" in line
    assert "reason=no-before-reading" in line
    assert "never a pass" in line.lower()


# ===========================================================================
# THE REFUSAL: a MISMATCH acts.  It is not a log line.
# ===========================================================================


def test_a_mismatch_is_a_named_refusal_not_a_log_line():
    mod = _mod()
    before = _reading(mod, "before", _inventory(seed=7))
    after = _reading(mod, "after", _inventory(seed=80))
    verdict = mod.compare(before, after)
    exc = verdict.refusal()
    assert isinstance(exc, mod.Weg2SeamDigestMismatch)
    assert mod.REFUSAL_MARKER in str(exc)
    assert mod.W_CODE == "W90"
    assert mod.REFUSAL_MARKER.startswith("W90 Weg2SeamDigestMismatch")
    # the refusal carries the localisation, not just the fact
    assert "pieces_moved=" in str(exc)
    # MATCH and UNARMED refuse nothing.
    assert mod.compare(before, _reading(mod, "after", _inventory(seed=7))).refusal() is None


def _code_strings(path):
    """Every string literal in a file that is NOT a docstring, with its line.

    THE #995 RULE, IN THE INSTRUMENT INSTEAD OF IN THE READER'S MEMORY: a
    W-code written in a COMMENT or a DOCSTRING is the code being TALKED ABOUT,
    never a second holder of it.  The first version of this scan counted the
    mint's own provenance comment ("W90 enumerated from ...") as a rival holder
    and went red on a tree with exactly one holder.  Strings survive the filter
    because ``W_CODE = "W90"`` and the launcher's help text are both real
    holders and both are strings.
    """
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return []
    docs = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docs.add(id(first.value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docs
    ]


def test_the_w_code_is_this_exception_and_no_other():
    """The #1263 census law, applied at the mint rather than after it."""
    _mod()
    holders = []
    for path in _py_files():
        for lineno, value in _code_strings(path):
            if re.search(r"\bW90\b", value):
                holders.append((os.path.relpath(path, REPO_ROOT), lineno, value))
    assert holders, "W90 is not written anywhere: the refusal was never minted"
    for rel, lineno, value in holders:
        assert "Weg2SeamDigestMismatch" in value or value.strip() == "W90", (
            f"W90 names a second holder at {rel}:{lineno}: {value!r}"
        )


# ===========================================================================
# AWAKE-ONLY: exactly two call sites, both on the awake side of one rank.
# ===========================================================================

_READING_CALLS = ("_weg2_seam_digest_before", "_weg2_seam_digest_after")


def _py_files():
    for root, _dirs, names in os.walk(
        os.path.join(REPO_ROOT, "python", "sglang", "srt")
    ):
        for n in names:
            if n.endswith(".py"):
                yield os.path.join(root, n)


def test_the_only_call_sites_are_the_two_awake_seams():
    """Mutant M4 (a third call site, at the cutover) dies here."""
    callers = {}
    for path in _py_files():
        with open(path, encoding="utf-8") as fh:
            body = fh.read()
        for hook in _READING_CALLS:
            if f"self.{hook}(" in body:
                callers.setdefault(hook, set()).add(os.path.relpath(path, REPO_ROOT))
    expected = {"python/sglang/srt/managers/scheduler_components/weight_updater.py"}
    assert callers.get(_READING_CALLS[0]) == expected, callers
    assert callers.get(_READING_CALLS[1]) == expected, callers


def test_no_cutover_or_flip_module_takes_a_reading():
    """The structural half of awake-only, stated as a walk rather than a claim.

    At flip time the sleeping group holds no weight bytes in VRAM at all
    (`understand_tensor-map.md` 5.4), so a reading there could only hash
    unmapped pages or an empty set.  There is nothing to open back up here.
    """
    forbidden = []
    for path in _py_files():
        rel = os.path.relpath(path, REPO_ROOT)
        if "flip" not in rel and "cutover" not in rel:
            continue
        with open(path, encoding="utf-8") as fh:
            if "seam_digest" in fh.read():
                forbidden.append(rel)
    assert forbidden == [], (
        f"a flip/cutover module references the seam digest: {forbidden}. "
        "The sleeping group holds no weight bytes at that moment; this is "
        "structurally impossible, not merely expensive."
    )


def test_the_before_reading_precedes_every_pause_of_the_weights_family():
    """The pages must still be mapped.  Anything read after a pause is a read
    of unmapped memory -- the campaign (a) fault, on the weights tag."""
    src = inspect.getsource(_wu().release_memory_occupation)
    i_read = src.index("self._weg2_seam_digest_before(")
    i_pause = src.index("self.memory_saver_adapter.pause(tag)")
    assert i_read < i_pause, (
        "the pre-pause reading runs AFTER a weights pause: it reads unmapped "
        "pages"
    )


def test_the_after_reading_follows_the_landing():
    """AFTER the family is complete, AFTER the reload, AFTER the shadow compare
    -- the same placement rule the shadow's destination hook already states and
    for the same reason: a reading before the bytes settle grades undefined
    content and its MISMATCH means nothing."""
    src = inspect.getsource(_wu().resume_memory_occupation)
    i_guard = src.index("if family_complete:")
    i_reload = src.index("self._weg2_wake_reload_weights()")
    i_read = src.index("self._weg2_seam_digest_after(")
    assert i_guard < i_reload < i_read
    line = [ln for ln in src.splitlines() if "self._weg2_seam_digest_after(" in ln][0]
    guard = [ln for ln in src.splitlines() if "if family_complete:" in ln][0]
    assert (len(line) - len(line.lstrip())) > (len(guard) - len(guard.lstrip()))


def test_the_wake_hook_raises_the_named_refusal():
    """Mutant M5 (log the MISMATCH and carry on) dies here.

    Asserted on the hook's own source rather than by execution, because the
    wake path needs a model runner, a process group and a device -- which is
    exactly how an unreachable call site stayed invisible for four boots
    (#1342 S1b).  What can be checked without a GPU is that the refusal is
    RAISED and not merely built.
    """
    src = inspect.getsource(_wu()._weg2_seam_digest_after)
    assert "raise " in src, "the wake hook builds a refusal it never raises"
    assert src.index("logger.info") < src.index("raise "), (
        "the verdict line must reach the log before the raise"
    )


def test_the_reading_is_rank_local_no_peer_no_collective():
    """Both moments are outside the no-return region and neither moves a byte
    between cards.  #875 DO-NOT-BUILD stays untouched, and that is a property
    of this module's own text."""
    src = inspect.getsource(_mod())
    for forbidden in (
        "torch.distributed",
        "all_gather",
        "all_reduce",
        "barrier(",
        "broadcast",
        "nccl",
        "gloo",
    ):
        assert forbidden not in src, (
            f"the seam digest reaches for {forbidden!r}: it is rank-local by "
            "construction and a collective here would put a hash on the "
            "no-return path"
        )


def test_the_host_buffer_is_bounded():
    """The host-RAM law again: a piece is hashed in bounded blocks, never
    assembled into a second image.  A default chunk larger than a layer would
    reintroduce the very term this whole slice exists to delete."""
    mod = _mod()
    assert 0 < mod.DEFAULT_CHUNK_BYTES <= (64 << 20)
    _geom, t = _inventory()[0]
    assert mod.piece_digest(t) == mod.piece_digest(t, chunk_bytes=1)


# ===========================================================================
# EXECUTION SMOKE OF THE PRODUCT CALL-SITES.
#
# desk-written-never-executed: every slice needs proof that its own code RAN,
# and the tests above are structural or module-level.  These drive the two
# HOOKS -- the real methods, on a real manager object, through the real
# producer (`card_inventory`) -- hermetically, CPU only, no device, no process
# group.  What they cannot drive is the pause/resume around them; that is the
# boot's job, and the placement pins above are what stand in for it.
# ===========================================================================


class _FakeServerArgs:
    enable_memory_saver = True
    enable_weights_cpu_backup = True
    enable_draft_weights_cpu_backup = False
    speculative_draft_model_path = None
    model_path = "/models/main"


class _FakeModel:
    def __init__(self, inventory):
        self._params = [
            (geom.name, tensor) for geom, tensor in inventory
        ]

    def named_parameters(self):
        return list(self._params)


class _FakeRunner:
    def __init__(self, model):
        self.model = model


class _FakeWorker:
    def __init__(self, model):
        self.model_runner = _FakeRunner(model)


class _Req:
    epoch = 11


@pytest.fixture()
def hooked(monkeypatch):
    """A manager whose model is the fixture inventory, on the real producer."""
    from sglang.srt.managers import weg2_memory_saver as ms
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    # The launcher's own channel for the ring layout, so `card_inventory`
    # resolves the family and the tags exactly as it does in a boot.
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_LAYERS, "8")
    monkeypatch.setenv(ms.WEIGHT_CHUNK_ENV_COUNT, "8")
    monkeypatch.setenv(_mod().ENV_ARM, "1")
    WU = wu.SchedulerWeightUpdaterManager
    monkeypatch.setattr(WU, "_weg2_group_name", lambda self: "P", raising=True)
    monkeypatch.setattr(WU, "_weg2_rank", lambda self: 1, raising=True)
    monkeypatch.setattr(WU, "_weg2_device_index", lambda self: 2, raising=True)
    monkeypatch.setattr(WU, "_weg2_server_args", lambda self: _FakeServerArgs(),
                        raising=True)

    def _make(inventory):
        return wu.SchedulerWeightUpdaterManager(
            tp_worker=_FakeWorker(_FakeModel(inventory)), draft_worker=None,
            tp_cpu_group=None, memory_saver_adapter=None,
            flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True,
        )

    return _make


def test_smoke_the_arming_line_is_announced_once_and_only_when_armed(
    hooked, monkeypatch, caplog
):
    """The arming line must have a CALLER, and exactly one announcement.

    An instrument's provenance line that nothing calls is the
    built-but-never-wired class; a line printed per flip is noise.  Both
    directions are asserted here because the fix for one is the defect for the
    other.
    """
    mod = _mod()
    mod.reset_announcement()
    m = hooked(_inventory())
    with caplog.at_level("INFO"):
        m._weg2_seam_digest_before(_Req(), ["weights_0"])
        m.weg2_seam_before = None
        m._weg2_seam_digest_before(_Req(), ["weights_0"])
    assert caplog.text.count("knob provenance") == 1, (
        "the arming line was announced 0 or >1 times"
    )
    assert mod.ARM_FLAG in caplog.text

    # UNARMED: nothing at all, because a serving boot's log gains nothing from
    # an instrument that is off.
    mod.reset_announcement()
    monkeypatch.delenv(mod.ENV_ARM, raising=False)
    caplog.clear()
    with caplog.at_level("INFO"):
        m._weg2_seam_digest_before(_Req(), ["weights_0"])
    assert "knob provenance" not in caplog.text


def test_smoke_the_hooks_run_and_a_clean_round_trip_is_a_match(hooked, caplog):
    """THE HAPPY PATH, EXECUTED.  Both hooks, the real producer, no device."""
    mod = _mod()
    inv = _inventory()
    m = hooked(inv)
    with caplog.at_level("INFO"):
        m._weg2_seam_digest_before(_Req(), ["weights_0", "weights"])
        assert m.weg2_seam_before is not None, "the pre-pause reading was not kept"
        assert m.weg2_seam_before.n_tensors > 0, (
            "the producer yielded no pieces: the smoke proved nothing"
        )
        # THE LANDING: the same values, a different arena -- which is what the
        # exchange does.  The manager's model is replaced, exactly as the
        # resume replaces the pages behind it.
        m.tp_worker = _FakeWorker(_FakeModel(_rearena(inv)))
        m._weg2_seam_digest_after(_Req(), ["weights_0", "weights"])
    assert m.weg2_seam_before is None, "the reading was not cleared after use"
    text = caplog.text
    assert f"verdict={mod.VERDICT_MATCH}" in text
    assert "stage=before" in text and "stage=after" in text
    assert mod.LIMIT_CLAUSE in text


def test_smoke_a_corrupted_landing_raises_the_named_refusal(hooked, caplog):
    """THE DANGER PATH, EXECUTED.  One element moves and the wake leg refuses.

    Mutant M5 turns the raise into a log line and this is what goes red -- by
    EXECUTION, not by reading the method's source.
    """
    mod = _mod()
    inv = _inventory()
    m = hooked(inv)
    m._weg2_seam_digest_before(_Req(), ["weights_0"])
    landed = _rearena(inv)
    landed[0][1][1, 1] = landed[0][1][1, 1] + 1.0
    m.tp_worker = _FakeWorker(_FakeModel(landed))
    with caplog.at_level("INFO"):
        with pytest.raises(mod.Weg2SeamDigestMismatch) as exc:
            m._weg2_seam_digest_after(_Req(), ["weights_0"])
    assert mod.REFUSAL_MARKER in str(exc.value)
    assert "qkv_proj" in str(exc.value), "the refusal did not name the piece"
    # the evidence reached the log BEFORE the raise
    assert f"verdict={mod.VERDICT_MISMATCH}" in caplog.text
    assert m.weg2_seam_before is None, (
        "a refusal left the pre-pause reading standing: the next wake would be "
        "graded against a landing that already failed"
    )


def test_smoke_the_unarmed_hooks_touch_nothing(hooked, monkeypatch, caplog):
    """Mutant M2 dies here too, and by EXECUTION on the product path.

    The manager is given a model that EXPLODES if anything walks it, so the
    assertion is "the unarmed boot did not read the weights", not merely "a
    boolean was False".
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    monkeypatch.delenv(_mod().ENV_ARM, raising=False)

    class _Exploder:
        def named_parameters(self):
            raise AssertionError("the unarmed hook walked the model")

    m = wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(_Exploder()), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )
    m.weg2_seam_before = "a stale reading from an armed flip"
    with caplog.at_level("INFO"):
        m._weg2_seam_digest_before(_Req(), ["weights_0"])
        assert m.weg2_seam_before is None, "the unarmed path INHERITED a reading"
        m._weg2_seam_digest_after(_Req(), ["weights_0"])
    assert _mod().LINE_PREFIX not in caplog.text, (
        "the unarmed grader still wrote a line"
    )


def test_smoke_an_unreadable_inventory_is_unarmed_and_never_raises(hooked, caplog):
    """The observer must not take a flip down because it could not read.

    An inventory the producer refuses is a NAMED absence on the line, never a
    MISMATCH -- the difference between "I could not grade" and "the bytes are
    wrong" is the whole reason UNARMED exists.
    """
    mod = _mod()
    m = hooked(_inventory())
    m._weg2_seam_digest_before(_Req(), ["weights_0"])
    m.tp_worker = _FakeWorker(None)  # the runner has no model any more
    with caplog.at_level("INFO"):
        m._weg2_seam_digest_after(_Req(), ["weights_0"])  # must not raise
    assert f"verdict={mod.VERDICT_UNARMED}" in caplog.text
    assert "no-inventory:no-model" in caplog.text


def test_an_absent_inventory_is_unarmed_never_a_match():
    """The green-by-vacancy direction: nothing to read is not a pass."""
    mod = _mod()
    empty = _reading(mod, "after", [])
    assert empty.n_tensors == 0
    verdict = mod.compare(None, empty)
    assert verdict.verdict == mod.VERDICT_UNARMED
    assert verdict.refusal() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
