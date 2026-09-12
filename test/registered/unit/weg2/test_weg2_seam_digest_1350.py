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
recomputed after the landing, same rank, same shard.  Both sides rank-own: no
peer, no collective, no byte movement between cards, both moments OUTSIDE the
no-return region (#875 DO-NOT-BUILD untouched).  Cost is a hash, not an image,
so the host-RAM law stays intact.  It is even SUPERIOR to the memcmp it
replaces: a memcmp against a second copy would have proven COPY fidelity; this
proves ROUND-TRIP fidelity, which is the question the exchange must answer.

THE THREE OPERATOR CONDITIONS ARE DESIGN CONDITIONS, NOT ADDENDA, and each has
its own test below plus a mutant that must die:

  (1) THE DIGEST COVERS THE LOGICAL TENSOR CONTENT, per rank, per shard, never
      an arena span, and it is FORM-KEYED.  The exchange changes the layout BY
      DESIGN (other tag, other arena), so a span digest would compare two
      different things and be Instrument-lies class A.
      -> `test_the_digest_survives_a_layout_change_that_preserves_content`
      -> `test_the_byte_count_is_the_logical_extent_not_the_storage_span`
      -> mutant M1 (span instead of content) dies on both.
  (2) DEFAULT OFF.  Armed only for instrument boots (S6I shape) through a NAMED
      knob.  Hashing ~9.6 GiB per rank is not free; always-on would be a flip
      cost regression, i.e. a performance defect the grader itself causes.
      -> `test_default_is_off_and_not_one_byte_is_read`
      -> mutant M2 (always-on) dies there.
  (3) AN HONEST LIMIT IN THE LOG LINE AND IN THE RECORD: a digest DETECTS, it
      does not LOCALIZE; a green digest is never "layout verified".
      -> `test_the_verdict_line_carries_the_honest_limit_verbatim`

AWAKE-ONLY IS BINDING, and it is a STRUCTURAL fact rather than a cost choice
(`understand_tensor-map.md` section 5.4 + ``weg2_memory_saver.py:123``,
``WEG2_SLEEP_TAGS = {kv_cache, weights}``): at flip time the sleeping group
holds NO weight bytes in VRAM at all.  A "compare at the cutover" design is not
expensive, it is IMPOSSIBLE.  The two moments are therefore: the last instant
before the pause, and the first instant after the landing -- both on the awake
side of the same rank.
  -> `test_the_only_call_sites_are_the_two_awake_seams`
  -> `test_no_cutover_or_flip_module_takes_a_reading`
  -> mutant M4 (a reading at the cutover) dies there.

THE DANGER DIRECTION NOBODY WOULD SEE: a comparison ACROSS TWO FORMS.  If the
form set moved, the content digests describe different things and comparing
them is not a finding.  The compare REFUSES by naming the form instead of
grading content.
  -> `test_a_changed_form_refuses_the_content_compare_by_name`
  -> mutant M3 (form_key ignored) dies there.

AND THE ONE THIS PROJECT KEEPS PAYING FOR: a verdict that is produced, logged
and then not acted upon.  A MISMATCH RAISES.
  -> `test_a_mismatch_is_a_named_refusal_not_a_log_line`
  -> `test_the_wake_hook_raises_the_named_refusal`
  -> mutant M5 (log, do not refuse) dies on both.

RED-FIRST STATE AT ``d294bde41d``: ``sglang.srt.weg2.seam_digest`` does not
exist, so every test below fails by NAME (the import is deferred into `_mod()`
for exactly that reason -- a module-level import would collapse the whole file
into one collection error and prove nothing per name).
"""

import importlib
import inspect
import os
import re

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def _mod():
    """The module under test, imported LATE so each test reds by its own name."""
    return importlib.import_module("sglang.srt.weg2.seam_digest")


def _wu():
    from sglang.srt.managers.scheduler_components import weight_updater

    return weight_updater.SchedulerWeightUpdaterManager


# ---------------------------------------------------------------------------
# Fixtures: a rank's shard, in two layouts that hold the SAME logical values.
# ---------------------------------------------------------------------------
def _shard(seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return [
        ("layer.0.qkv_proj.weight", torch.randn(64, 40, generator=g).to(torch.bfloat16)),
        ("layer.0.mlp.down_proj.weight", torch.randn(40, 96, generator=g)),
        ("layer.0.input_layernorm.weight", torch.randn(40, generator=g)),
        ("embed_tokens.weight", torch.randn(128, 40, generator=g).to(torch.bfloat16)),
    ]


def _relaid(named):
    """The SAME logical values, in a deliberately different physical layout.

    Every property a span digest could key on is moved and the content is not:

    * a FRESH allocation (other arena, other ``data_ptr``),
    * a non-zero ``storage_offset``,
    * a transposed, NON-CONTIGUOUS stride,
    * a storage whose byte extent is far larger than the tensor's own.

    This is the harder half of the red-first falsifier, and the half that
    proves the digest is not accidentally a layout digest.
    """
    out = []
    for name, t in named:
        if t.dim() == 1:
            big = torch.zeros(t.shape[0] + 11, dtype=t.dtype)
            view = big[5 : 5 + t.shape[0]]
        else:
            big = torch.zeros(t.shape[1] + 3, t.shape[0] + 5, dtype=t.dtype)
            view = big[3:, 5:].t()
        view.copy_(t)
        out.append((name, view))
    return out


def _reading(mod, stage, named, *, tags=("weights",), rank=1, epoch=4):
    return mod.take_reading(
        stage,
        named,
        group="P",
        rank=rank,
        tags=tags,
        epoch=epoch,
    )


# ===========================================================================
# CONDITION (1): content, not span.  Form-keyed.
# ===========================================================================


def test_the_digest_survives_a_layout_change_that_preserves_content():
    """THE HARD HALF OF THE RED-FIRST FALSIFIER, and condition (1) itself.

    The exchange changes the layout by design.  A digest that moves when only
    the layout moved would report MISMATCH on every single correct flip -- the
    instrument would be the defect.  Mutant M1 (hash the storage span) dies
    here.
    """
    mod = _mod()
    before = _reading(mod, "before", _shard(), tags=("weights_0", "weights_1"))
    after = _reading(mod, "after", _relaid(_shard()), tags=("weights",))
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MATCH, (
        "a layout change that preserves content was graded "
        f"{verdict.verdict}/{verdict.reason}: this is a layout digest, not a "
        "content digest"
    )
    assert before.digest == after.digest


def test_the_byte_count_is_the_logical_extent_not_the_storage_span():
    """Second face of M1: even the reported ``bytes`` must be the logical one.

    The relaid shard sits in a storage several times its own size.  A reading
    that priced the span would print a number no reader could reconcile with
    the shard, and would be the same category error one level down.
    """
    mod = _mod()
    named = _shard()
    logical = sum(t.numel() * t.element_size() for _, t in named)
    assert _reading(mod, "before", named).bytes == logical
    assert _reading(mod, "after", _relaid(named)).bytes == logical


def test_a_single_flipped_byte_after_the_landing_is_a_mismatch():
    """THE OTHER HALF OF THE FALSIFIER: the direction the grader exists for."""
    mod = _mod()
    before = _reading(mod, "before", _shard())
    landed = _relaid(_shard())
    # ONE element of ONE shard, in the middle, on the relaid side.
    name, t = landed[1]
    t[3, 7] = t[3, 7] + 1.0
    after = _reading(mod, "after", landed)
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MISMATCH
    assert verdict.reason == mod.REASON_CONTENT


def test_the_digest_is_not_a_count_or_a_name_check():
    """DENOMINATOR FIRST: a digest that hashed only names and shapes would pass
    both tests above and detect nothing.  Same forms, different content."""
    mod = _mod()
    a = _reading(mod, "before", _shard(seed=7))
    b = _reading(mod, "after", _shard(seed=8))
    assert a.form_key == b.form_key, "the two shards must share one form"
    assert a.digest != b.digest
    assert mod.compare(a, b).verdict == mod.VERDICT_MISMATCH


# ===========================================================================
# THE FORM KEY: a comparison across two forms is refused, never graded.
# ===========================================================================


def test_a_changed_form_refuses_the_content_compare_by_name():
    """Mutant M3 (ignore ``form_key``) dies here.

    Note what is asserted: not merely that the verdict is not MATCH -- a
    content compare across two forms also fails to match, so that assertion
    would survive M3 -- but that the REASON names the form.  Grading content
    across two forms is not a weaker finding, it is a different one.
    """
    mod = _mod()
    before = _reading(mod, "before", _shard())
    moved = _shard()
    moved[0] = (moved[0][0], moved[0][1][:32])  # the shard's own form changed
    after = _reading(mod, "after", moved)
    verdict = mod.compare(before, after)
    assert verdict.verdict == mod.VERDICT_MISMATCH
    assert verdict.reason == mod.REASON_FORM, (
        "a form change was graded as a content change: the compare read "
        "content across two forms"
    )
    assert before.form_key != after.form_key
    line = verdict.line()
    assert f"form_key_before={before.form_key}" in line
    assert f"form_key_after={after.form_key}" in line


def test_the_form_key_ignores_layout_and_moves_with_shape_or_dtype():
    mod = _mod()
    named = _shard()
    assert mod.form_key(named) == mod.form_key(_relaid(named))
    other = _shard()
    other[2] = (other[2][0], other[2][1].to(torch.float64))
    assert mod.form_key(named) != mod.form_key(other)


# ===========================================================================
# CONDITION (2): default OFF, armed by a NAMED knob through the #901 authority.
# ===========================================================================


class _Exploding(list):
    """An iterable that refuses to be read.  ~9.6 GiB per rank is the cost this
    guards; a test that only checked a boolean could not tell an unarmed hook
    from an armed hook whose result was thrown away."""

    def __iter__(self):
        raise AssertionError(
            "the unarmed path read the shard: the digest is not default-off"
        )


def test_default_is_off_and_not_one_byte_is_read():
    """Mutant M2 (always-on) dies here."""
    mod = _mod()
    assert mod.seam_digest_armed({}) is False
    assert mod.take_reading_if_armed(
        "before", _Exploding(), group="P", rank=0, tags=(), epoch=1, environ={}
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
    assert "seam_digest.ENV_ARM" in src or "SGLANG_WEG2_SEAM_DIGEST" in src
    assert "--weg2-seam-digest" in src


# ===========================================================================
# CONDITION (3): the honest limit, in the line and in the record.
# ===========================================================================


def test_the_verdict_line_carries_the_honest_limit_verbatim():
    mod = _mod()
    before = _reading(mod, "before", _shard())
    after = _reading(mod, "after", _relaid(_shard()))
    line = mod.compare(before, after).line()
    assert mod.LIMIT_CLAUSE in line
    assert "DETECTS" in mod.LIMIT_CLAUSE and "LOCALIZE" in mod.LIMIT_CLAUSE


def test_the_verdict_line_carries_every_field_the_boot_seat_reads():
    mod = _mod()
    before = _reading(mod, "before", _shard(), tags=("weights_0",), rank=2, epoch=9)
    after = _reading(mod, "after", _relaid(_shard()), tags=("weights",), rank=2, epoch=10)
    line = mod.compare(before, after).line()
    assert line.startswith(mod.LINE_PREFIX)
    for field in (
        "rank=2",
        "group=P",
        "tag=",
        "n_tensors=4",
        f"bytes={before.bytes}",
        f"digest_before={before.digest}",
        f"digest_after={after.digest}",
        f"verdict={mod.VERDICT_MATCH}",
        "reason=",
        "population=",
    ):
        assert field in line, f"the verdict line has no {field!r}: {line}"


def test_unarmed_is_a_named_absence_and_never_reads_as_a_pass():
    """UNARMED is what a grader that produced NO verdict says.  It must be
    distinguishable from MATCH by anything that reads the line, and it must
    never raise either -- an absence is not a finding in either direction."""
    mod = _mod()
    v = mod.unarmed_verdict("no-before-reading", group="P", rank=0, tags=("weights",))
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
    before = _reading(mod, "before", _shard(seed=7))
    after = _reading(mod, "after", _shard(seed=8))
    verdict = mod.compare(before, after)
    exc = verdict.refusal()
    assert isinstance(exc, mod.Weg2SeamDigestMismatch)
    assert mod.REFUSAL_MARKER in str(exc)
    assert mod.W_CODE == "W90"
    assert mod.REFUSAL_MARKER.startswith("W90 Weg2SeamDigestMismatch")
    # MATCH and UNARMED refuse nothing.
    assert mod.compare(before, _reading(mod, "after", _shard(seed=7))).refusal() is None


def test_the_w_code_is_this_exception_and_no_other():
    """The #1263 census law, applied at the mint rather than after it."""
    mod = _mod()
    hits = set()
    for root, _dirs, names in os.walk(os.path.join(REPO_ROOT, "python", "sglang", "srt")):
        for n in names:
            if not n.endswith(".py"):
                continue
            path = os.path.join(root, n)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if re.search(r"\bW90\b", line):
                        hits.add(re.sub(r"\s+", " ", line).strip())
    assert hits, "W90 is not written anywhere: the refusal was never minted"
    for line in hits:
        assert "Weg2SeamDigestMismatch" in line or "W_CODE" in line, (
            f"W90 names a second holder: {line}"
        )


# ===========================================================================
# AWAKE-ONLY: exactly two call sites, both on the awake side of one rank.
# ===========================================================================

_READING_CALLS = ("_weg2_seam_digest_before", "_weg2_seam_digest_after")


def _py_files():
    for root, _dirs, names in os.walk(os.path.join(REPO_ROOT, "python", "sglang", "srt")):
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
    expected = "python/sglang/srt/managers/scheduler_components/weight_updater.py"
    assert callers.get(_READING_CALLS[0]) == {expected}, callers
    assert callers.get(_READING_CALLS[1]) == {expected}, callers


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
            body = fh.read()
        if "seam_digest" in body:
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
    line = [l for l in src.splitlines() if "self._weg2_seam_digest_after(" in l][0]
    guard = [l for l in src.splitlines() if "if family_complete:" in l][0]
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
    i_raise = src.index("raise ")
    i_log = src.index("logger.info")
    assert i_log < i_raise, "the verdict line must reach the log before the raise"


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
    """The host-RAM law again: the shard is hashed in bounded blocks, never
    assembled into a second image.  A default chunk larger than a layer would
    reintroduce the very term this whole slice exists to delete."""
    mod = _mod()
    assert 0 < mod.DEFAULT_CHUNK_BYTES <= (64 << 20)
    named = _shard()
    # A chunk smaller than one row must not change the answer.
    full = mod.content_digest(named)
    tiny = mod.content_digest(named, chunk_bytes=1)
    assert full == tiny


def test_an_absent_shard_is_unarmed_never_a_match():
    """The green-by-vacancy direction: nothing to read is not a pass."""
    mod = _mod()
    empty = _reading(mod, "after", [])
    assert empty.n_tensors == 0
    verdict = mod.compare(None, empty)
    assert verdict.verdict == mod.VERDICT_UNARMED
    assert verdict.refusal() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
