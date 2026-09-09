# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#1292: P and D collide on one phase-footprint dump filename.

ROOT CAUSE (see WEG2_BUILD_DECISIONS_0906.md SECTION 1an-S3, weg2sb5g,
2026-09-09): ``activation_probe._global_rank()``
(python/sglang/srt/mem_ledger/activation_probe.py:260-281) returns a rank
that is unique only WITHIN one ``torch.distributed`` process group. Weg-2's
P (pp_size=3, tp_size=1) and D (tp_size=3, pp_size=1) are two independently
launched jobs that share one dump directory whenever both are armed with
the identical ``SGLANG_PHASE_FOOTPRINT_DUMP`` -- the current recipe does
exactly that. Both wrote ``phase_footprint_rank{0,1,2}.json``; D wrote
first, P wrote later and silently overwrote every one of D's three dumps
(``write_footprint_dump``'s ``os.replace(tmp, path)`` at line 155 was
unconditional). BOOT_weg2sb5g_0909.md's gate 10 read "PASS -- 3 files
written" while all three surviving files carried P's profile alone.

THE FIX, three parts, ONE mechanism (the profile digest already computed
for the on-disk payload -- ``profile_digest_from_canonical``, split out of
``sglang.srt.mem_ledger.activation.profile_key`` -- reused rather than a
second identity invented):

1. ``activation_probe.dump_filename()`` folds the Weg-2 group tag into the
   filename. The tag is ``SGLANG_WEG2_GROUP`` read via
   ``weg2_memory_saver.weg2_group_name()`` -- the one thing in the tree
   that already tells a rank which Weg-2 group it is in
   (``weg2/launcher.py``'s ``build_env(group="P"|"D")``) -- not a new
   identity. Outside Weg-2 the tag is ``""`` and the filename is
   byte-identical to the pre-fix shape.
2. ``write_footprint_dump()`` refuses (does not overwrite) a same-name
   write whose profile digest differs from the file already on disk --
   residual protection for whatever future path still collides.
3. ``scripts/vram_ledger/probe_activation.py``'s ``ingest`` groups dumps by
   profile digest and prints/writes one verdict block PER GROUP, so a
   directory holding both groups' dumps folds into two calibrations,
   never one merged (or one blanket-refused) calibration.

Hermetic: no driver, no CUDA, no torch device -- ``write_footprint_dump``
and ``ingest`` are pure file/JSON code once armed.
"""

import importlib.util
import json
import logging
import os

from sglang.srt.mem_ledger import activation_probe as ap
from sglang.srt.mem_ledger.activation import (
    ActivationProfile,
    load_footprints,
    profile_digest_from_canonical,
)

_HERE = os.path.abspath(__file__)
_ROOT = _HERE
for _ in range(5):
    _ROOT = os.path.dirname(_ROOT)
SCRIPT = os.path.join(_ROOT, "scripts", "vram_ledger", "probe_activation.py")
assert os.path.isfile(SCRIPT), SCRIPT

FP = "a191a0712717"

#: Structurally different from each other on exactly the fields that make
#: P (pure PP) and D (pure TP) two different activation profiles -- the
#: reason ingest must never fold them into one calibration.
PROFILE_P = ActivationProfile(
    architectures=("Qwen3_8ForCausalLM",),
    chunked_prefill_size=4096,
    tp_size=1,
    pp_size=3,
)
PROFILE_D = ActivationProfile(
    architectures=("Qwen3_8ForCausalLM",),
    chunked_prefill_size=4096,
    tp_size=3,
    pp_size=1,
)


def load_script():
    spec = importlib.util.spec_from_file_location("probe_activation_1292", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(rank, group, profile, uuid, activation_mib, capture_mib, dump_dir):
    return ap.write_footprint_dump(
        rank=rank,
        card_uuid=uuid,
        hw_fingerprint=FP,
        profile_canonical=profile.canonical(),
        activation_peak_bytes=(16000 + activation_mib) << 20,
        capture_bytes=capture_mib << 20,
        reserved_peak_bytes=(16000 + activation_mib + 300) << 20,
        prefill_tokens=1234,
        dump_dir=str(dump_dir),
        peak_floor_bytes=16000 << 20,
        group=group,
    )


# ---------------------------------------------------------------------------
# RED 1: two groups, one directory -> distinct files
# ---------------------------------------------------------------------------


def test_two_groups_same_rank_same_dir_write_distinct_files(tmp_path):
    path_p = _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, tmp_path)
    path_d = _write(0, "D", PROFILE_D, "GPU-d0", 850, 640, tmp_path)

    assert path_p and path_d
    assert path_p != path_d, (
        "P and D rank 0 wrote the SAME filename -- the #1292 collision "
        "reproduced"
    )
    assert os.path.exists(path_p) and os.path.exists(path_d)

    with open(path_p) as f:
        dp = json.load(f)
    with open(path_d) as f:
        dd = json.load(f)
    assert dp["group"] == "P" and dd["group"] == "D"
    assert dp["profile"] == PROFILE_P.canonical()
    assert dd["profile"] == PROFILE_D.canonical()


def test_ungrouped_boot_keeps_the_pre_fix_filename_shape(tmp_path):
    """No SGLANG_WEG2_GROUP -> group="" -> byte-identical to the old shape,
    so a non-Weg2 boot's dump filenames are unaffected by this fix."""
    path = _write(2, "", PROFILE_P, "GPU-x", 900, 640, tmp_path)
    assert os.path.basename(path) == "phase_footprint_rank2.json"


# ---------------------------------------------------------------------------
# RED 2: same filename, different profile digest -> refusal, not overwrite
# ---------------------------------------------------------------------------


def test_same_name_different_profile_is_refused_not_overwritten(tmp_path, caplog):
    path1 = _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, tmp_path)
    with open(path1) as f:
        original = json.load(f)

    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        path2 = _write(0, "P", PROFILE_D, "GPU-p0", 111, 22, tmp_path)

    assert path2 is None, (
        "a same-name write carrying a DIFFERENT profile must refuse, not "
        "silently overwrite"
    )
    with open(path1) as f:
        after = json.load(f)
    assert after == original, "a refused write must leave the existing dump untouched"
    assert any("PHASE-FOOTPRINT REFUSED" in r.message for r in caplog.records)
    assert any(
        "W18 Weg2PhaseFootprintCollision" in r.message for r in caplog.records
    )


def test_same_name_same_profile_rewrite_is_the_normal_running_peak_path(tmp_path):
    """The refusal must not break record_prefill_peak's own "keep the
    running peak" rewrite: same group, same rank, same profile, a new
    (higher) peak -- this must still overwrite in place."""
    path1 = _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, tmp_path)
    path2 = _write(0, "P", PROFILE_P, "GPU-p0", 950, 700, tmp_path)
    assert path1 == path2
    with open(path2) as f:
        d = json.load(f)
    assert d["activation_delta_bytes"] == 950 << 20


# ---------------------------------------------------------------------------
# RED 3: ingest on a mixed-group directory -> two verdict blocks, two
# cache entries, never one merged calibration
# ---------------------------------------------------------------------------


def test_ingest_on_a_mixed_group_dir_produces_two_verdict_blocks(tmp_path, capsys):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    cache = tmp_path / "cache"
    _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, dumps)
    _write(1, "P", PROFILE_P, "GPU-p1", 880, 700, dumps)
    _write(0, "D", PROFILE_D, "GPU-d0", 850, 640, dumps)
    _write(1, "D", PROFILE_D, "GPU-d1", 820, 640, dumps)

    m = load_script()
    assert m.ingest(str(dumps), str(cache)) == 0

    out = capsys.readouterr().out
    assert out.count("Wrote 2 card footprint(s)") == 2, out

    got_p = load_footprints(hw_fingerprint=FP, profile=PROFILE_P, cache_dir=str(cache))
    got_d = load_footprints(hw_fingerprint=FP, profile=PROFILE_D, cache_dir=str(cache))
    assert set(got_p) == {"GPU-p0", "GPU-p1"}
    assert set(got_d) == {"GPU-d0", "GPU-d1"}
    assert got_p["GPU-p0"].activation_mib == 900
    assert got_d["GPU-d0"].activation_mib == 850


def test_load_dumps_glob_widens_to_group_qualified_filenames(tmp_path):
    """The glob that used to be ``phase_footprint_rank*.json`` must also see
    the group-qualified shape, or a mixed directory is read HALF-blind."""
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, dumps)
    _write(0, "D", PROFILE_D, "GPU-d0", 850, 640, dumps)
    m = load_script()
    loaded = m.load_dumps(str(dumps))
    assert len(loaded) == 2, loaded


# ---------------------------------------------------------------------------
# MUTANTS -- the danger direction each RED test exists to catch
# ---------------------------------------------------------------------------


def test_mutant_dropping_the_group_tag_reintroduces_the_collision(tmp_path):
    """MUTANT: ``dump_filename`` ignores ``group`` and always returns
    ``phase_footprint_rank{rank}.json`` -- exactly the pre-#1292 shape.
    Reproduced by hand here (not by calling the real function) to show the
    filenames it would produce for P rank 0 and D rank 0 collide, which is
    the root cause SECTION 1an-S3 found. The real ``dump_filename`` (used
    above in every RED test) does not collide.
    """

    def mutant_dump_filename(rank, group=""):  # group silently ignored
        return f"phase_footprint_rank{rank}.json"

    assert mutant_dump_filename(0, "P") == mutant_dump_filename(0, "D"), (
        "the mutant collides P and D on rank 0 -- this is the #1292 bug"
    )
    assert ap.dump_filename(0, "P") != ap.dump_filename(0, "D")


def test_mutant_silent_overwrite_would_destroy_the_other_groups_dump(tmp_path):
    """MUTANT: skip the profile-digest check and go back to the pre-#1292
    unconditional ``os.replace(tmp, path)``. Demonstrated by hand: writing
    D's dump then unconditionally replacing the same path with P's payload
    (what the old code did) leaves D's measurement unrecoverable. The real
    ``write_footprint_dump`` (same filename, forced via an identical group)
    refuses instead and D's original dump survives -- see
    ``test_same_name_different_profile_is_refused_not_overwritten`` above,
    repeated here framed as the mutant it defeats.
    """
    path_d = _write(0, "D", PROFILE_D, "GPU-d0", 850, 640, tmp_path)
    with open(path_d) as f:
        d_payload = json.load(f)

    # The mutant: what os.replace(tmp, path) unconditionally did before #1292.
    mutant_payload = {**d_payload, "group": "P", "profile": PROFILE_P.canonical()}
    tmp_file = path_d + ".mutant_tmp"
    with open(tmp_file, "w") as f:
        json.dump(mutant_payload, f)
    os.replace(tmp_file, path_d)
    with open(path_d) as f:
        clobbered = json.load(f)
    assert clobbered["group"] == "P", (
        "the mutant clobbers D's dump with P's -- exactly what P's later "
        "boot did to D on weg2sb5g"
    )

    # Restore D's dump, as if the mutant had never run, then exercise the
    # REAL function against the same collision: it must refuse.
    with open(path_d, "w") as f:
        json.dump(d_payload, f)
    refused = _write(0, "D", PROFILE_P, "GPU-d0", 111, 22, tmp_path)
    assert refused is None
    with open(path_d) as f:
        assert json.load(f) == d_payload, "the real code must leave D's dump intact"


def test_mutant_merging_groups_at_ingest_attributes_one_groups_peak_to_the_other(
    tmp_path,
):
    """MUTANT: ingest treats every dump in the directory as one profile
    (the shape ``ingest`` had before #1292's grouping -- pick the FIRST
    dump's profile and fold every card into that one calibration).
    Reproduced by hand against the real dump payloads to show it would
    attribute D's activation numbers to P's profile digest. The real
    ``ingest`` (grouped by digest, exercised in
    ``test_ingest_on_a_mixed_group_dir_produces_two_verdict_blocks`` above)
    never does this.
    """
    dumps_dir = tmp_path / "dumps"
    dumps_dir.mkdir()
    _write(0, "P", PROFILE_P, "GPU-p0", 900, 700, dumps_dir)
    _write(0, "D", PROFILE_D, "GPU-d0", 850, 640, dumps_dir)

    m = load_script()
    raw_dumps = m.load_dumps(str(dumps_dir))
    assert len(raw_dumps) == 2

    # The mutant: everything folds under the FIRST dump's profile digest.
    mutant_digest = profile_digest_from_canonical(raw_dumps[0]["profile"])
    mutant_groups = {mutant_digest: raw_dumps}
    assert len(mutant_groups) == 1, (
        "the mutant merges P and D into one profile group -- D's card would "
        "be reserved for under P's activation profile"
    )

    # The real grouping keeps them apart.
    real_groups = {}
    for d in raw_dumps:
        digest = d.get("profile_digest") or profile_digest_from_canonical(
            d.get("profile")
        )
        real_groups.setdefault(digest, []).append(d)
    assert len(real_groups) == 2, real_groups
