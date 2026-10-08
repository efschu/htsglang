"""LS6 (30.09.): six 27B Leistungsschalter default-on IN THE CODE after their metal proof.

User rule 29.09. ~10:15Z: proven on metal -> default on in the code, "off" only until the first
series. Metal: z30y5m dkr27browauthoritybar1fs09301128 (cfdb9b50b3; 15/15, group death 0, needle
MATCH), checked by /spinning/gpu-arb/docker/pending/ls12-0930/check_ls12.py -> GRUEN for:

  (1) --p-trim-end-anchor        'PDFLIP P-TRIM-END-ANCHOR n=.. tokens=N->N-1'  12
  (2) --p-host-overlap           '#PGAP ... overlap=1'                        756/756
  (3) --p-prefill-graph 512      'PREFILL-GRAPH captured backend=full'        3/3
  (6) FLLIPER_DCP_LSE_MERGE=a2a   'DCP-MERGE-BLOCK ... mode=a2a'               3/3
  (11) FLLIPER_PDFLIP_DC_OFF_PATH   'PDFLIP-DC-OFFPATH epoch='                     11
  (12) FLLIPER_PDFLIP_QUIESCE_FAST  'PDFLIP-QUIESCE-FAST group='                   13

Per switch: the qwen27b row turns it on for an unset flag/env; an explicit value (flag, --no- twin,
env 0/ar) wins; nextflash and "no form" keep the code default (NF decides its own).
"""

import pytest

from flliper.srt.environ import envs
from flliper.srt.pdflip import form as FM

INT8 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
GGUF = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-GGUF-unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf"
NVFP4 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return FM.PdFlipForm(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                       flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(FM.FORM_ENV, raising=False)
    for n in ("FLLIPER_PDFLIP_DC_OFF_PATH", "FLLIPER_PDFLIP_QUIESCE_FAST", "FLLIPER_DCP_LSE_MERGE"):
        monkeypatch.delenv(n, raising=False)
    return monkeypatch


def _as(clean, profile):
    if profile is None:
        clean.delenv(FM.FORM_ENV, raising=False)
    else:
        clean.setenv(FM.FORM_ENV, _form_env(profile))


def _defaults(profile, argv=(), model=None):
    from flliper.srt.pdflip import launcher as L

    words = ["--tree", "/t", "--tag", "x"]
    if profile:
        words += ["--profile", profile]
    if model is not None:
        words += ["--model", model]
    words += list(argv)
    ns = L.build_parser().parse_args(words)
    L.apply_profile_arg_defaults(ns, words)
    return ns


# --- the registry rows ------------------------------------------------------

def test_rows_carry_the_six():
    q, n = FM.PROFILES["qwen27b"], FM.PROFILES["nextflash"]
    assert q.end_anchor == "trim" and n.end_anchor != "trim"
    assert (q.p_host_overlap, n.p_host_overlap) == (True, False)
    assert (q.p_prefill_graph, n.p_prefill_graph) == (512, 0)
    assert set(q.p_prefill_graph_formats) == {"int8", "nvfp4", "fp8"}
    assert (q.d_dcp_lse_merge, n.d_dcp_lse_merge) == ("a2a", "ar")
    assert (q.front_dc_off_path, n.front_dc_off_path) == (True, False)
    assert (q.front_quiesce_fast, n.front_quiesce_fast) == (True, False)
    assert FM.PROFILE_SWITCH_DEFAULTS["qwen27b"]["FLLIPER_DCP_LSE_MERGE"] == "a2a"
    assert FM.PROFILE_SWITCH_DEFAULTS["nextflash"]["FLLIPER_DCP_LSE_MERGE"] == "ar"


# --- (1) --p-trim-end-anchor ------------------------------------------------

@pytest.mark.parametrize("profile,argv,want", [
    ("qwen27b", (), True),
    (None, (), True),                                     # --profile default = qwen27b
    ("qwen27b", ("--no-p-trim-end-anchor",), False),      # A/B arm wins
    ("qwen27b", ("--p-trim-end-anchor",), True),
    ("nextflash", (), False),                             # NF end_anchor=tail_handoff
])
def test_1_trim_end_anchor(profile, argv, want):
    assert _defaults(profile, argv).p_trim_end_anchor is want


# --- (2) --p-host-overlap ---------------------------------------------------

@pytest.mark.parametrize("profile,argv,want", [
    ("qwen27b", (), True),
    ("qwen27b", ("--no-p-host-overlap",), False),
    ("qwen27b", ("--p-host-overlap",), True),
    ("nextflash", (), False),
])
def test_2_host_overlap(profile, argv, want):
    assert _defaults(profile, argv).p_host_overlap is want


def test_2_host_overlap_env_reaches_group_p():
    from flliper.srt.pdflip import launcher as L

    ns = _defaults("qwen27b")
    env = L.p_host_overlap_env(ns.p_host_overlap, False)
    assert env.get("FLLIPER_PDFLIP_P_HOST_OVERLAP") == "1"


# --- (3) --p-prefill-graph 512 ----------------------------------------------

@pytest.mark.parametrize("profile,argv,model,want", [
    ("qwen27b", (), INT8, 512),
    ("qwen27b", (), NVFP4, 512),
    ("qwen27b", (), GGUF, 0),                              # 27b-gguf.env runs without a graph
    ("qwen27b", ("--p-prefill-graph", "0"), INT8, 0),      # graphcalE-style arm wins
    ("qwen27b", ("--p-prefill-graph=1024",), INT8, 1024),
    ("nextflash", (), None, 0),
])
def test_3_prefill_graph(profile, argv, model, want):
    assert _defaults(profile, argv, model).p_prefill_graph == want


def test_3_graph_default_keeps_the_chunk_policy_runnable():
    """The chunk policy's runnability reads the graph bucket: with the graph now a
    row default, the INT8 row gets dynamic WITHOUT any profile flag (rc9j form)."""
    ns = _defaults("qwen27b", (), INT8)
    assert (ns.p_prefill_graph, ns.p_chunk_policy) == (512, "dynamic")


# --- (6) FLLIPER_DCP_LSE_MERGE=a2a -------------------------------------------

def _merge_mode():
    from flliper.srt.layers.dcp import comm as C

    C._LSE_MERGE["mode"] = None
    try:
        return C.lse_merge_mode()
    finally:
        C._LSE_MERGE["mode"] = None


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, "a2a"),
    ("qwen27b", "", "a2a"),            # blank = unset
    ("qwen27b", "ar", "ar"),           # explicit wins
    ("nextflash", None, "ar"),
    (None, None, "ar"),
    ("nextflash", "a2a", "a2a"),
])
def test_6_dcp_lse_merge(clean, profile, explicit, want):
    _as(clean, profile)
    if explicit is not None:
        clean.setenv("FLLIPER_DCP_LSE_MERGE", explicit)
    assert _merge_mode() == want


# --- (11) FLLIPER_PDFLIP_DC_OFF_PATH -------------------------------------------

@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),
    ("qwen27b", "", True),
    ("qwen27b", "0", False),
    ("nextflash", None, False),
    (None, None, False),
    ("nextflash", "1", True),
])
def test_11_dc_off_path(clean, profile, explicit, want):
    from flliper.srt.pdflip import front as F

    _as(clean, profile)
    if explicit is not None:
        clean.setenv("FLLIPER_PDFLIP_DC_OFF_PATH", explicit)
    assert F._env_switch_on_or_profile(F.DC_OFF_PATH_ENV) is want


def test_11_front_reads_it_through_the_row(clean):
    from flliper.srt.pdflip import front as F

    _as(clean, "qwen27b")
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="ls6", store_dir="/tmp",
                prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1)
    assert f._dc_off_path is True
    assert "dc_off_path=on" in f.flipfast_line()


# --- (12) FLLIPER_PDFLIP_QUIESCE_FAST ------------------------------------------

@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, True),
    ("qwen27b", "0", False),
    ("nextflash", None, False),
    (None, None, False),
])
def test_12_quiesce_fast(clean, profile, explicit, want):
    from flliper.srt.pdflip import front as F

    _as(clean, profile)
    if explicit is not None:
        clean.setenv("FLLIPER_PDFLIP_QUIESCE_FAST", explicit)
    assert envs.FLLIPER_PDFLIP_QUIESCE_FAST.get() is want
    interval, fast = F.quiesce_poll_s()
    assert (interval, fast) == ((0.01, True) if want else (0.05, False))
