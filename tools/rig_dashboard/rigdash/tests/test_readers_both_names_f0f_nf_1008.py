"""F0-F NF (08.10.): two readers of the dashboard that took the legacy spelling literally.

* ``profile_recompute.env_map`` hands the planner modules the group envs of a profile.  The renamed ``profile_couplings`` looks up
  ``FLLIPER_MOE_RESIDENT_EXPERT_FRACTION``; the live NF profile (/spinning/gpu-arb/docker/profiles/nf-int4-h6-abl.env, written before the
  rename) still carries the legacy spelling, so the Balken of that profile lost the resident fraction (RED in
  test_profil_balken_aph2_1006::test_the_real_abl_profile_yields_the_reference_group_lines before this change).  The keys now come back
  in the renamed spelling; when both spellings stand in one text the renamed one wins.
* ``kartenplan_build.records.boot_logs``: the record builder globbed ``boot_<old>_<tag>_*`` only and found no log of a boot of the renamed tree.

No old token is spelled in one piece here (the rename tool must not rewrite these expectations).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from kartenplan_build import records as REC  # noqa: E402
from rigdash import names as N  # noqa: E402
from rigdash import profile_recompute as R  # noqa: E402

LEG = "SG" "LANG_"       # legacy runtime prefix
SUB = "WE" "G2_"
OLD = "we" "g2"
NEW = "pdflip"


def test_canonical_env_name_maps_the_runtime_families_only():
    assert N.canonical_env_name(LEG + "MOE_RESIDENT_EXPERT_FRACTION") == "FLLIPER_MOE_RESIDENT_EXPERT_FRACTION"
    assert N.canonical_env_name(LEG + SUB + "OWNED_CUT_X1") == "FLLIPER_PDFLIP_OWNED_CUT_X1"
    assert N.canonical_env_name(LEG + "OPT_" + SUB + "D_SEAT_VRAM") == "FLLIPER_OPT_PDFLIP_D_SEAT_VRAM"
    for same in ("FLLIPER_X", "FLLIPER_PDFLIP_X", "HTS" "GLANG_INSTRUMENTS", "SGL_FOO", "PATH", LEG, ""):
        assert N.canonical_env_name(same) == same


def test_env_map_hands_the_planner_modules_the_renamed_key():
    m = R.env_map(LEG + "MOE_RESIDENT_EXPERT_FRACTION=0.06,0.51;FLLIPER_MOE_SCRATCH_SLOTS=4;" + LEG + SUB + "OWNED_CUT_X1=workers")
    assert m == {"FLLIPER_MOE_RESIDENT_EXPERT_FRACTION": "0.06,0.51", "FLLIPER_MOE_SCRATCH_SLOTS": "4", "FLLIPER_PDFLIP_OWNED_CUT_X1": "workers"}
    # both spellings in one text: the renamed one wins, in either order
    assert R.env_map(LEG + "X=old;FLLIPER_X=new") == {"FLLIPER_X": "new"}
    assert R.env_map("FLLIPER_X=new;" + LEG + "X=old") == {"FLLIPER_X": "new"}
    assert R.env_map("A=1;B=0.1,0.2;;C=x=y;bare") == {"A": "1", "B": "0.1,0.2", "C": "x=y"}


def test_the_record_builder_finds_the_boot_logs_of_either_generation(tmp_path):
    ev = str(tmp_path)
    for name in ("boot_%s_tagold_abc_0929_1.front.log" % OLD, "boot_%s_tagnew_abc_1008_1.front.log" % NEW, "boot_%s_tagnew_abc_1008_1.D.log" % NEW):
        open(os.path.join(ev, name), "w").close()
    base = lambda tag, suf: [os.path.basename(p) for p in REC.boot_logs(ev, tag, suf)]      # noqa: E731
    assert base("tagold", ".front.log") == ["boot_%s_tagold_abc_0929_1.front.log" % OLD]
    assert base("tagnew", ".front.log") == ["boot_%s_tagnew_abc_1008_1.front.log" % NEW]
    assert base("tagnew", ".D.log") == ["boot_%s_tagnew_abc_1008_1.D.log" % NEW]
    assert base("tagnew", ".P.log") == [] and base("nope", ".D.log") == []

