"""27B Leistungsschalter, second pass (30.09.): the switches every 27B profile
still had to set although their metal proof exists. User rule 29.09. ~10:15Z
(memory leistungsschalter-nach-nachweis-default-an-0929): a switch proven on
metal goes default-on IN THE CODE; the docker profile line becomes redundant.

Group 1 -- THE HG BASE (D host gap, 27B row 24h): the four switches the dhg
measurement ran UNDER in both arms (profiles/27b-int8-dhg.env: "Form =
27b.env (u.a. FLLIPER_PDFLIP_D_DEFER_SEQ_LENS_CPU=1 + FLLIPER_PDFLIP_D_DEFER_REBUILD=1,
FLLIPER_DFLASH_PLAN_SYNC_FREE=1, FLLIPER_DFLASH_WINDOW_POOL_SYNC_FREE=1) plus
genau die Schalter unten"). The HG levers are default-on for qwen27b since
df762c4bbc, but FLLIPER_PDFLIP_D_EARLY_DRAFT is stage 3 of the deferred length
read and INERT without FLLIPER_PDFLIP_D_DEFER_SEQ_LENS_CPU -- the registry named
an on-switch that could not act without the profile.

Per switch: qwen27b ON without env, explicit 0 wins, nextflash unchanged (off),
no form unchanged (off).
"""

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import form as FM


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


# ---------------------------------------------------------------------------
# Group 1: the HG base
# ---------------------------------------------------------------------------

HG_BASE = (
    "FLLIPER_PDFLIP_D_DEFER_SEQ_LENS_CPU",
    "FLLIPER_PDFLIP_D_DEFER_REBUILD",
    "FLLIPER_DFLASH_PLAN_SYNC_FREE",
    "FLLIPER_DFLASH_WINDOW_POOL_SYNC_FREE",
)


def _base_readers():
    """Each switch asked the way its rank-side caller asks it."""
    from flliper.srt.managers import pdflip_d_hostgap as H

    return {
        "FLLIPER_PDFLIP_D_DEFER_SEQ_LENS_CPU": H.defer_seq_lens_cpu_on,
        "FLLIPER_PDFLIP_D_DEFER_REBUILD": H.defer_rebuild_on,
        "FLLIPER_DFLASH_PLAN_SYNC_FREE": lambda: bool(envs.FLLIPER_DFLASH_PLAN_SYNC_FREE.get()),
        "FLLIPER_DFLASH_WINDOW_POOL_SYNC_FREE":
            lambda: bool(envs.FLLIPER_DFLASH_WINDOW_POOL_SYNC_FREE.get()),
    }


def test_hg_base_is_one_registry_field():
    assert tuple(FM.HG_BASE_SWITCHES) == HG_BASE
    assert FM.PROFILES["qwen27b"].d_hostgap_base is True
    assert FM.PROFILES["nextflash"].d_hostgap_base is False
    for name in HG_BASE:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][name] is True
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][name] is False


def test_every_row_with_hg_levers_carries_their_base():
    """EARLY_DRAFT is inert without the deferred read: a row that turns the
    levers on without the base names an on-switch that cannot act."""
    for pid, prof in FM.PROFILES.items():
        if prof.d_hostgap_levers:
            assert prof.d_hostgap_base, pid


@pytest.mark.parametrize("name", HG_BASE)
@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "", True),        # blank = unset
    ("qwen27b", "0", False),      # explicit off wins
    ("qwen27b", "1", True),
    ("nextflash", None, False),   # NF unchanged: off (its D is MTP)
    ("nextflash", "1", True),     # an explicit on still reaches NF
    (None, None, False),          # no form: the code default (off), unchanged
])
def test_hg_base_default(clean, name, profile, explicit, want):
    for n in HG_BASE:
        clean.delenv(n, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(name, explicit)
    assert _base_readers()[name]() is want


def test_early_draft_acts_on_the_27b_row_without_any_env(clean):
    """The point of group 1: with nothing set, the 27B row arms the deferred
    read AND the early draft on top of it (before: early draft armed, deferral
    off -> inert)."""
    from flliper.srt.managers import pdflip_d_hostgap as H

    for n in HG_BASE + ("FLLIPER_PDFLIP_D_EARLY_DRAFT",):
        clean.delenv(n, raising=False)
    _as(clean, "qwen27b")
    assert H.early_draft_on() is True
    assert H.defer_seq_lens_cpu_on() is True
    assert H.defer_rebuild_on() is True


# ---------------------------------------------------------------------------
# Group 2: the two release-draft BUG FIXES (27b-release-draft.env _form, every
# 27B boot since rc10/rc11), one registry field ``d_release_fixes``:
#  * FLLIPER_DFLASH_WINDOW_POOL_DEDUP_CARRY (DK 7faa13dc1c): radix dedup freed
#    fresh draft slots -> ~2046 unmapped draft prefix slots per round and
#    request, 12.1 M per rank in i8h (D logs on d98b3ba08a); armed on metal
#    in every 27B boot since ('DEDUP-CARRY armed' 3/3, 0929_175011).
#  * FLLIPER_PDFLIP_CENSUS_O1_EVICT (KR e54ac95c65): n4h dkr27bnvfp4bar1mwh09261131
#    240k P->D flip 9.5 s, 6.3 s of it the kv resume unread behind the
#    quadratic FIFO evict at the 524288-key cap. Witness on metal under agent
#    load: w109290020 'KR CENSUS-O1-EVICT ledger at its cap (524288 keys)'
#    3/3 ranks at 00:51:00 inside the P->D resume of epoch 150 (done
#    00:50:58,740 -> first content 00:51:01,224 = 2.5 s, no 6 s stall).
# ---------------------------------------------------------------------------

FIXES = ("FLLIPER_DFLASH_WINDOW_POOL_DEDUP_CARRY", "FLLIPER_PDFLIP_CENSUS_O1_EVICT")


def _fix_readers():
    from flliper.srt.mem_cache import producer_phase_census as C

    def census():
        C._o1_evict = None  # read once per process: reset the cache per case
        return C.census_o1_evict_armed()

    return {
        "FLLIPER_DFLASH_WINDOW_POOL_DEDUP_CARRY":
            lambda: bool(envs.FLLIPER_DFLASH_WINDOW_POOL_DEDUP_CARRY.get()),
        "FLLIPER_PDFLIP_CENSUS_O1_EVICT": census,
    }


def test_release_fixes_are_one_registry_field():
    assert tuple(FM.RELEASE_FIX_SWITCHES) == FIXES
    assert FM.PROFILES["qwen27b"].d_release_fixes is True
    assert FM.PROFILES["nextflash"].d_release_fixes is False
    for name in FIXES:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][name] is True
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][name] is False


@pytest.mark.parametrize("name", FIXES)
@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),
    ("qwen27b", "", True),
    ("qwen27b", "0", False),
    ("qwen27b", "1", True),
    ("nextflash", None, False),   # NF decides its own default (KR pick e17bd548b5)
    ("nextflash", "1", True),
    (None, None, False),
])
def test_release_fixes_default(clean, name, profile, explicit, want):
    for n in FIXES:
        clean.delenv(n, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(name, explicit)
    try:
        assert _fix_readers()[name]() is want
    finally:
        from flliper.srt.mem_cache import producer_phase_census as C

        C._o1_evict = None
