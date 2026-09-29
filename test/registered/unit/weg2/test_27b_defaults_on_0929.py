"""27B performance switches ON BY DEFAULT after their metal proof (user rule
29.09. ~10:15Z: a switch proven on metal goes default-on IN THE CODE, not only
in the docker profile). Source: gpu-arb/docs/LEISTUNGSSCHALTER-INVENTAR-0929.md,
section "27B", list (a).

Per switch, three facts:
  * the qwen27b row turns it ON without any env / flag,
  * an explicit off (env ``0`` / the CLI flag) wins,
  * the nextflash row stays what it was (NF byte-identical).
"""

import pytest

from sglang.srt.environ import envs
from sglang.srt.weg2 import form as FM


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    for k in (FM.FORM_ENV,):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


# ---------------------------------------------------------------------------
# 1. SGLANG_WEG2_FRONT_EXACT_TOKENS -- w109290020 X-EXACT-TOKENS 108x match=1
# ---------------------------------------------------------------------------

XE = "SGLANG_WEG2_FRONT_EXACT_TOKENS"


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "0", False),      # explicit off wins
    ("nextflash", None, True),    # NF unchanged (on since V1 27.09.)
    ("nextflash", "0", False),
    (None, None, False),          # no form: the code fallback, unchanged
])
def test_front_exact_tokens_default(clean, profile, explicit, want):
    clean.delenv(XE, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(XE, explicit)
    assert envs.SGLANG_WEG2_FRONT_EXACT_TOKENS.get() is want
    assert FM.PROFILES["qwen27b"].front_exact_tokens is True
    assert FM.PROFILES["nextflash"].front_exact_tokens is True


# ---------------------------------------------------------------------------
# 2. HG bundle (row 24h): SGLANG_WEG2_D_EARLY_DRAFT + SGLANG_DFLASH_ACCEPT_SYNC_
#    FUSED + SGLANG_BARLINK_BAR1_CANON_ORDER, measured only together --
#    dkr27bint8dhgbar1dhg109261456 vs dkr27bbar1i8h109261444: step time better
#    at all 24 points (10k bs1 code -2.9 %, prose -4.4 %, 240k -3.4/-2.3 %).
#    ONE registry field (d_hostgap_levers); nextflash off (CANON_ORDER would
#    change NF's bar1 reduction order -- the inventory does not prove it on NF).
# ---------------------------------------------------------------------------

HG = ("SGLANG_WEG2_D_EARLY_DRAFT", "SGLANG_DFLASH_ACCEPT_SYNC_FUSED",
      "SGLANG_BARLINK_BAR1_CANON_ORDER")


def _hg_readers():
    """The three rank-side readers, each asked the way its caller asks."""
    from sglang.srt.distributed.device_communicators import barlink_bar1 as B
    from sglang.srt.managers import weg2_d_hostgap as H
    from sglang.srt.speculative import dflash_worker_v2 as W

    return {
        "SGLANG_WEG2_D_EARLY_DRAFT": H.early_draft_on,
        "SGLANG_DFLASH_ACCEPT_SYNC_FUSED": W.accept_sync_fused_on,
        "SGLANG_BARLINK_BAR1_CANON_ORDER": B.canon_order_on,
    }


def test_hg_bundle_is_one_registry_field():
    assert FM.PROFILES["qwen27b"].d_hostgap_levers is True
    assert FM.PROFILES["nextflash"].d_hostgap_levers is False
    for name in HG:
        assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][name] is True
        assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][name] is False


@pytest.mark.parametrize("name", HG)
@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),      # default ON on the 27B row
    ("qwen27b", "0", False),      # explicit off wins
    ("qwen27b", "1", True),
    ("nextflash", None, False),   # NF unchanged: off
    ("nextflash", "1", True),     # an explicit on still reaches NF
    (None, None, False),          # no form: the code default (off), unchanged
])
def test_hg_bundle_default(clean, name, profile, explicit, want):
    for n in HG:
        clean.delenv(n, raising=False)
    _as(clean, profile)
    if explicit is not None:
        clean.setenv(name, explicit)
    assert _hg_readers()[name]() is want
    assert getattr(envs, name).get() is want


# ---------------------------------------------------------------------------
# 3. REGISTRY -> CLI DEFAULT (one mechanism: launcher PROFILE_ARG_DEFAULTS /
#    apply_profile_arg_defaults, UNIFY S3) and its first best-form field:
#    --d-replayssm-spec from ModelProfile.replayssm. 27B xsn436: D-KV +21/+13 %,
#    not slower (user 24.09. ~20:47Z); NF x172/x174 (its own inventory (a)).
# ---------------------------------------------------------------------------

import argparse  # noqa: E402


def _parsed(argv):
    from sglang.srt.weg2 import launcher as L

    words = ["--tree", "/t", "--tag", "x"] + list(argv)
    return L.build_parser().parse_args(words), words


def _defaults(profile, argv=(), model=None):
    from sglang.srt.weg2 import launcher as L

    extra = ["--profile", profile] if profile else []
    if model is not None:
        extra += ["--model", model]
    ns, words = _parsed(extra + list(argv))
    L.apply_profile_arg_defaults(ns, words)
    return ns


def test_replayssm_parser_default_is_the_code_default_off():
    from sglang.srt.weg2 import launcher as L

    ns, _ = _parsed([])
    assert ns.d_replayssm_spec == L.D_REPLAYSSM_SPEC_DEFAULT == "off"


@pytest.mark.parametrize("profile,argv,want", [
    ("qwen27b", (), "on"),                                   # row replayssm=True
    (None, (), "on"),                                        # --profile default = qwen27b
    ("qwen27b", ("--d-replayssm-spec", "off"), "off"),       # explicit off wins
    ("qwen27b", ("--d-replayssm-spec=off",), "off"),
    ("nextflash", ("--d-replayssm-spec", "on"), "on"),       # every NF profile passes on: byte-equal
    ("nextflash", ("--d-replayssm-spec", "off"), "off"),
])
def test_replayssm_cli_default_follows_the_row(profile, argv, want):
    assert _defaults(profile, argv).d_replayssm_spec == want


def test_replayssm_nextflash_row_is_its_measured_form():
    """nextflash's row says replayssm=True (NF inventory (a): x172/x174 TP0
    -396 MiB, 257k needle MATCH; every NF profile passes --d-replayssm-spec on),
    so its CLI default follows the row like the 27B's."""
    assert FM.PROFILES["nextflash"].replayssm is True
    assert _defaults("nextflash").d_replayssm_spec == "on"


def test_the_one_mechanism_names_every_registry_flag():
    from sglang.srt.weg2 import launcher as L

    flags = [f for f, _d, _o in L.PROFILE_ARG_DEFAULTS]
    assert "--d-replayssm-spec" in flags
    assert len(flags) == len(set(flags))


# ---------------------------------------------------------------------------
# 4. --d-token-placement bandwidth (27B row 24b) -- INT8 only: rc9meas INT8
#    with R, depth gain -1.1 ... -3.1 % at 128k/240k; NVFP4 no gain, FP8/GGUF
#    unmeasured -> capacity. --d-reshard stays off (inventory point 7, class b).
# ---------------------------------------------------------------------------

_MC = "/spinning/llm_stuff/club-3090/models-cache/"
INT8 = _MC + "Qwen3.8-27B-INT8-gdncov-vocabembed"
FP8 = _MC + "Qwen3.8-27B-FP8"
NVFP4 = _MC + "Qwen3.8-27B-NVFP4-RadixArk"
GGUF = _MC + "Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf"
NF_INT4 = _MC + "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


@pytest.mark.parametrize("profile,model,argv,want", [
    ("qwen27b", INT8, (), "bandwidth"),                                  # 27B INT8: default on
    (None, None, (), "bandwidth"),                                       # launcher defaults = 27B INT8
    ("qwen27b", INT8, ("--d-token-placement", "capacity"), "capacity"),  # explicit wins
    ("qwen27b", FP8, (), "capacity"), ("qwen27b", GGUF, (), "capacity"),  # unmeasured formats
    ("qwen27b", NVFP4, (), "capacity"),                                  # measured, no gain
    ("nextflash", NF_INT4, (), "capacity"),                              # NF unchanged
])
def test_token_placement_cli_default_follows_the_row(profile, model, argv, want):
    assert _defaults(profile, argv, model=model).d_token_placement == want


def test_token_placement_rows():
    assert FM.PROFILES["qwen27b"].d_token_placement == "bandwidth"
    assert FM.PROFILES["qwen27b"].d_token_placement_formats == ("int8",)
    assert FM.PROFILES["nextflash"].d_token_placement == "capacity"
    assert FM.PROFILES["nextflash"].d_token_placement_formats == ()


def test_d_reshard_stays_off_by_default():
    """Inventory point 7 (wake-seg + drq) is class b: no registry default."""
    from sglang.srt.weg2 import launcher as L

    assert _defaults("qwen27b", (), model=INT8).d_reshard == L.D_RESHARD_DEFAULT == "off"
    assert "--d-reshard" not in [f for f, _d, _o in L.PROFILE_ARG_DEFAULTS]


# ---------------------------------------------------------------------------
# 5. MAMBA: registry = the metal form (inventory 27B, contradiction 1). Every
#    27B profile since 24.09. set SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE=1
#    (27b.env): the environ alias that ARMS the END-anchor carrier hold, and on
#    group P the inner-anchor release (c255e10ddb). The row said
#    mamba_carrier_hold=False -- a form no 27B boot ran.
# ---------------------------------------------------------------------------

HOLD = "SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD"
RELEASE = "SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE"


def test_mamba_rows_name_the_metal_form():
    assert FM.PROFILES["qwen27b"].mamba_carrier_hold is True
    assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][HOLD] is True
    assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"][RELEASE] is True
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][HOLD] is True      # NF unchanged
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"][RELEASE] is False  # NF unchanged


@pytest.mark.parametrize("profile,hold,alias,want_hold,want_release", [
    ("qwen27b", None, None, True, True),     # the 27B form without a profile line
    ("qwen27b", None, "1", True, True),      # the 27B profile line: same form
    ("qwen27b", None, "0", False, False),    # explicit alias off wins both halves
    ("qwen27b", "0", None, False, True),     # explicit hold off wins the hold only
    ("nextflash", None, None, True, False),  # NF unchanged
    (None, None, None, True, False),         # no form: unchanged (NF default hold, no release)
])
def test_mamba_hold_and_inner_release_defaults(clean, profile, hold, alias, want_hold, want_release):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    for k in (HOLD, RELEASE):
        clean.delenv(k, raising=False)
    clean.setenv("SGLANG_WEG2_GROUP", "P")
    clean.setattr(urc, "_WEG2_END_ANCHOR", True)
    _as(clean, profile)
    if hold is not None:
        clean.setenv(HOLD, hold)
    if alias is not None:
        clean.setenv(RELEASE, alias)
    assert envs.SGLANG_WEG2_ENABLE_MAMBA_CARRIER_HOLD.get() is want_hold
    assert urc._weg2_inner_anchor_release_on() is want_release


def test_inner_release_stays_group_p_only(clean):
    from sglang.srt.mem_cache import unified_radix_cache as urc

    clean.delenv(RELEASE, raising=False)
    clean.setattr(urc, "_WEG2_END_ANCHOR", True)
    _as(clean, "qwen27b")
    clean.setenv("SGLANG_WEG2_GROUP", "D")
    assert urc._weg2_inner_anchor_release_on() is False
