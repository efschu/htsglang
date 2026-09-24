"""H51 (Nutzer-Order 24.09. 14:15Z): "schreiben wir in D auch draft context in
den hicache? das muesste raus, weil draft ja keinen hicacheplatz mehr bekommt."

NF-form fixtures for the draft-tier switch ``SGLANG_WEG2_HICACHE_DRAFT_TIER``
(default ``auto`` = off without a draft-KV producer), built on the 27B line and
picked onto this one. These pin the NEXT-FLASH shape of boot fnFL2x158
(c01951e3e1), not the 27B one:

* D, --speculative-draft-placement solo: TP0 is the solo draft host (MTP,
  2 kv heads x 256 x fp8 = 1024 B/token), TP1/TP2 are Form-A shadows;
* page_size 64, arena host (SGLANG_HICACHE_ARENA_HOST=1, ARENA_GIB=4,
  KV page 786432 B) -> the paged draft pool is the PLAIN pinned class sized to
  the arena id space: x158 TP0 "HiCache draft KV registered: MHATokenToKVPool
  (host 353664 slots)" after "Allocating 0.36 GB pinned host memory";
* P boots without a drafter (H25, SGLANG_WEG2_DRAFT_ON_P=False on both groups),
  so nobody produces draft KV: 21/21 D PUBLISH-SWEEP draft_issued=0, 5/5 D
  admissions "draft pages not in the store", "#993 draft L3 READ: 2184
  page(s) requested, 0 hit".

H53 (after the pick of 27B 9eef037e8d): the 27B form resolves ``auto`` IN THE
LAUNCHER (p_group_has_draft_producer, NF form: H25's SGLANG_WEG2_DRAFT_ON_P)
and hands the RESOLVED ``off`` to both groups' environment; a rank reads only
``off`` (unset/auto/on = the tier as before, it cannot know P's form). So the
rank cases below run under the environment the launcher writes for the x158
form (``launcher_rank_env``), and the log texts are the 27B ones:
rank ``WEG2 HICACHE-DRAFT-TIER off (SGLANG_WEG2_HICACHE_DRAFT_TIER=off) at boot
registration``, launcher ``WEG2 HICACHE-DRAFT-TIER: off (auto: group P has no
draft producer)``, admission ``(draft tier armed=False)``.

RED BEFORE THE PICK (the tier is armed on c01951e3e1): the ``auto``/``off``
cases, the registered env, the disarm line, the host-restored scrub, the
ledger. GREEN BEFORE AND AFTER: explicit ``on`` (byte-identical registration),
a producer on P, a non-Weg-2 instance, group P, the device-hit warm guard.
"""

import logging
import types

import pytest
import torch

try:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.phase_flip_draft_bootstrap import (
        arm_draft_cold_for_admission,
        rounds_owed,
    )
    from sglang.srt.mem_cache import kv_cache_builder as kb
    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
        HybridCacheController,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.pool_host import mha as mha_host
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: conftest: this module pins its own draft form per case (SGLANG_WEG2_DRAFT_ON_P).
H25_OWN_DRAFT_FORM = True

TIER_ENV = "SGLANG_WEG2_HICACHE_DRAFT_TIER"
DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino"
#: every consume point of the one gate (cache_controller + hybrid overrides)
DIRECTIONS = ("write", "load", "l3-load", "l3-write", "admission")
#: x158: 4096 staging rows + 5461 arena slots x 64, page-rounded (+64)
X158_DRAFT_HOST_SLOTS = 353664


class PinnedDraftHostPool:
    """Stands in for the pinned host pool class: every construction IS a pinned
    allocation (x158: 0.36 GB cudaHostAlloc on D TP0) -- or, on the 27B page-1
    form, the draft arena bind. Counted, never allocated."""

    built: list = []

    def __init__(self, device_pool, host_to_device_ratio, host_size, page_size,
                 layout, *args, **kwargs):
        tokens = int(device_pool.size * host_to_device_ratio)
        self.page_size = int(page_size)
        self.size = (tokens // self.page_size + 1) * self.page_size
        self.layer_num = 1
        PinnedDraftHostPool.built.append(self)


class Ctl:
    """The controller's draft surface with the REAL methods bound (the stand-in
    discipline of test_draft_tier_gate_861)."""

    def __init__(self):
        self.has_draft = False
        self.solo_draft_shadow = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        self.draft_owner_phase = None
        self.draft_binding_generation = None
        self.draft_identity = None
        self.draft_total_kv_heads = None
        self._draft_disarm_warned = set()
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        # D TP0's KV host pool on x158: the arena anchor, 4096 staging rows
        self.mem_pool_host = types.SimpleNamespace(
            size=4096, arena_read=True, size_per_token=12288, page_size=64)

    set_draft_kv_pool = HiCacheController.set_draft_kv_pool
    disarm_draft_kv_pool = HiCacheController.disarm_draft_kv_pool
    draft_tier_armed = HiCacheController.draft_tier_armed
    _warn_draft_disarmed = HiCacheController._warn_draft_disarmed
    _maybe_register_draft_with_storage = HiCacheController._maybe_register_draft_with_storage
    _draft_component_name = HiCacheController._draft_component_name
    draft_claim_packed = HiCacheController.draft_claim_packed
    _draft_presence_transfer = HybridCacheController._draft_presence_transfer


class _Args(types.SimpleNamespace):
    """ServerArgs carries every field; an unknown one reads as None."""

    def __getattr__(self, name):
        return None


def nf_d_server_args():
    return _Args(
        enable_multi_layer_eagle=False,
        hicache_mem_layout="layer_first",
        hicache_storage_backend="file",
        speculative_algorithm="NEXTN",
        speculative_draft_model_path=DRAFT,
        speculative_draft_model_revision=None,
        speculative_draft_placement="solo",
        speculative_draft_kv_only=False,
        draft_kv_layout="replicated",
        page_size=64,
        kv_cache_dtype="fp8_e4m3",
        enable_hierarchical_cache=True,
        hicache_write_policy="write_back",
        hicache_io_backend="direct",
    )


class _Spec:
    def is_ngram(self):
        return False

    def is_none(self):
        return False

    def is_eagle(self):
        return True


def nf_draft_device_pool():
    pool = object.__new__(MHATokenToKVPool)
    pool.size = 262144
    pool.page_size = 64
    pool.head_num = 2
    pool.head_dim = 256
    pool.v_head_dim = 256
    pool.layer_num = 1
    return pool


def nf_draft_worker(*, shadow: bool):
    runner = types.SimpleNamespace(
        token_to_kv_pool=None if shadow else nf_draft_device_pool(),
        is_draft_solo_shadow=shadow,
        model_config=types.SimpleNamespace(get_total_num_kv_heads=lambda: 2),
    )
    return types.SimpleNamespace(draft_worker=types.SimpleNamespace(draft_runner=runner))


@pytest.fixture(autouse=True)
def _spy(monkeypatch):
    PinnedDraftHostPool.built = []
    monkeypatch.setattr(mha_host, "get_mha_host_pool_cls",
                        lambda device_pool, role="kv": PinnedDraftHostPool)
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_HOST", "1")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", "4")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_KV_PAGE_BYTES", "786432")
    monkeypatch.delenv(TIER_ENV, raising=False)
    yield


def launcher_rank_env(monkeypatch, *, draft_on_p):
    """What the NF launcher hands BOTH groups for the tier (build_env: pop,
    then hicache_draft_tier_env()), from H25's resolved draft_on_p as main
    installs it beside resolve_draft_on_p (HiCache enabled). Returns the
    launcher line."""
    from sglang.srt.weg2 import launcher as L

    monkeypatch.setitem(L._SPEC_FORM, "draft_kv_on_p", bool(draft_on_p))
    monkeypatch.delenv(TIER_ENV, raising=False)
    line = L.hicache_draft_tier_line()
    for k, v in L.hicache_draft_tier_env().items():
        monkeypatch.setenv(k, v)
    return line


def form(monkeypatch, *, group, draft_on_p, tier=None):
    """The rank environment of one group, as the x158 launcher wrote it
    (WEG2-GROUP-ENV: SGLANG_WEG2_GROUP=D, SGLANG_WEG2_DRAFT_ON_P=False).
    ``tier="launcher"``: the tier value the launcher resolves and writes."""
    if group is None:
        monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_GROUP", group)
    if draft_on_p is None:
        monkeypatch.delenv("SGLANG_WEG2_DRAFT_ON_P", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_DRAFT_ON_P", "True" if draft_on_p else "False")
    if tier == "launcher":
        return launcher_rank_env(monkeypatch, draft_on_p=bool(draft_on_p))
    if tier is not None:
        monkeypatch.setenv(TIER_ENV, tier)
    return None


def register(*, shadow: bool = False, draft_worker="nf"):
    ctl = Ctl()
    dw = nf_draft_worker(shadow=shadow) if draft_worker == "nf" else draft_worker
    kb.maybe_register_hicache_draft(
        tree_cache=types.SimpleNamespace(cache_controller=ctl),
        draft_worker=dw,
        spec_algorithm=_Spec(),
        server_args=nf_d_server_args(),
        enable_hierarchical_cache=True,
        page_size=64,
    )
    return ctl


def assert_tier_off(ctl):
    assert PinnedDraftHostPool.built == [], "a draft host pool was built (pinned / arena bind)"
    assert ctl.mem_pool_host_draft is None
    for d in DIRECTIONS:
        assert ctl.draft_tier_armed(d) is False, f"gate open for {d!r}"
    assert ctl.draft_page_get_func is None and ctl.draft_page_set_func is None
    assert ctl._draft_presence_transfer() is None  # no DRAFT-PRESENCE probe


def assert_tier_as_x158(ctl):
    """c01951e3e1's registration, byte for byte in what the log shows."""
    assert len(PinnedDraftHostPool.built) == 1
    assert PinnedDraftHostPool.built[0].size == X158_DRAFT_HOST_SLOTS
    assert ctl.mem_pool_host_draft is PinnedDraftHostPool.built[0]
    assert ctl.draft_owner_phase is None and ctl.draft_binding_generation is None
    assert ctl.draft_identity == kb.drafter_identity_hash(nf_d_server_args())
    assert ctl.draft_total_kv_heads == 2
    for d in DIRECTIONS:
        assert ctl.draft_tier_armed(d) is True


# --------------------------------------------------------- RED before the pick


def test_env_is_registered_with_default_auto():
    from sglang.srt.environ import envs

    field = getattr(envs, TIER_ENV, None)
    assert field is not None, f"{TIER_ENV} is not an Envs field (env-var-conventions rule 1)"
    assert str(field.get()).strip().lower() == "auto"


def test_nf_d_host_auto_without_producer_has_no_tier(monkeypatch):
    """x158 D TP0: 0.36 GB pinned (353664 slots x 1024 B = 345 MiB), gate open
    on all consume points, a DRAFT-PRESENCE probe per admission -- for draft
    KV nobody writes. The launcher reads the missing producer off H25's
    SGLANG_WEG2_DRAFT_ON_P=0 and hands D ``off``."""
    line = form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    assert line.startswith("WEG2 HICACHE-DRAFT-TIER: off (auto: group P has no draft producer)")
    import os

    assert os.environ.get(TIER_ENV) == "off"
    assert_tier_off(register())


def test_nf_launcher_reads_the_producer_off_h25_not_the_cli_default(monkeypatch):
    """Before main resolved it the launcher reads SGLANG_WEG2_DRAFT_ON_P (H25
    default 0), never --draft-kv-on-p's CLI default ``on``; the H25 A/B arm
    DRAFT_ON_P=1 keeps the tier and writes nothing into the groups."""
    from sglang.srt.weg2 import launcher as L

    monkeypatch.setitem(L._SPEC_FORM, "draft_kv_on_p", None)
    monkeypatch.delenv("SGLANG_WEG2_DRAFT_ON_P", raising=False)
    assert L.p_group_has_draft_producer() is False
    assert L.hicache_draft_tier_env() == {TIER_ENV: "off"}
    monkeypatch.setenv("SGLANG_WEG2_DRAFT_ON_P", "1")
    assert L.p_group_has_draft_producer() is True
    assert L.hicache_draft_tier_env() == {}
    monkeypatch.setenv(TIER_ENV, "off")
    with pytest.raises(L.Weg2LaunchRefused, match="W152 Weg2HicacheDraftTierConflict"):
        L.hicache_draft_tier_env()


def test_nf_no_993_draft_l3_read_under_tier_off(monkeypatch, caplog):
    """x158: '#993 draft L3 READ: 2184 page(s) requested, 0 hit' per flip,
    136 MiB zero fill. The one read site sits behind draft_tier_armed('l3-load')
    and the reader is installed only by set_draft_kv_pool: under off neither."""
    form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    ctl = register()
    assert ctl.draft_tier_armed("l3-load") is False
    assert ctl.draft_page_get_func is None
    with caplog.at_level(logging.INFO):
        assert HiCacheController._draft_page_get_flags(ctl, ["h"] * 3, list(range(192))) is None
    assert "#993 draft L3 READ" not in caplog.text


def test_nf_explicit_off_wins_over_a_producer(monkeypatch):
    form(monkeypatch, group="D", draft_on_p=True, tier="off")
    assert_tier_off(register())


def test_nf_solo_group_answers_one_scalar_claim_form(monkeypatch):
    """x158: 24/24 PREFETCH-CLAIM-FORM local_packed=True on TP0 AND the two
    shadows. The shadow mark exists only to mirror the host's packed draft
    form (xsn392); with the tier off there is nothing to mirror, so every rank
    answers scalar LOCALLY -- a host-only disarm makes the group agree packed
    on every prefetch and the host print agreed!=local each time."""
    form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    ranks = [register(), register(shadow=True), register(shadow=True)]
    assert [c.solo_draft_shadow for c in ranks] == [False, False, False]
    forms = [HiCacheController.draft_claim_packed(c) for c in ranks]
    assert forms == [False, False, False], forms


def test_nf_disarm_is_named_in_the_boot_log(monkeypatch, caplog):
    form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    with caplog.at_level(logging.INFO):
        register()
        register(shadow=True)
    lines = [r.getMessage() for r in caplog.records]
    named = [m for m in lines if m.startswith(
        "WEG2 HICACHE-DRAFT-TIER off (SGLANG_WEG2_HICACHE_DRAFT_TIER=off) at boot registration: "
        "the draft gets no HiCache space")]
    assert len(named) == 2, lines  # every D rank names it, host and shadow alike
    assert not any("HiCache draft KV registered" in m for m in lines)


# ------------------------------------------------ GREEN before and after the pick


def test_nf_explicit_on_is_the_x158_registration(monkeypatch):
    form(monkeypatch, group="D", draft_on_p=False, tier="on")
    ctl = register()
    assert_tier_as_x158(ctl)
    assert HiCacheController.draft_claim_packed(register(shadow=True)) is True


def test_nf_auto_with_the_mtp_producer_on_p_keeps_the_tier(monkeypatch):
    """The H25 A/B arm SGLANG_WEG2_DRAFT_ON_P=1: P carries the MTP head as
    the draft-KV producer, so D must read what it writes."""
    form(monkeypatch, group="D", draft_on_p=True)
    assert_tier_as_x158(register())


def test_non_weg2_instance_auto_keeps_the_tier(monkeypatch):
    """No Weg-2 group, DRAFT_ON_P unset: an ordinary speculating HiCache
    server is its own producer -- the upstream default must not move."""
    form(monkeypatch, group=None, draft_on_p=None)
    assert_tier_as_x158(register())


@pytest.mark.parametrize("tier", [None, "launcher"])
def test_nf_group_p_without_a_drafter_builds_nothing(monkeypatch, caplog, tier):
    form(monkeypatch, group="P", draft_on_p=False, tier=tier)
    with caplog.at_level(logging.INFO):
        ctl = register(draft_worker=None)
    assert_tier_off(ctl)
    assert ctl.solo_draft_shadow is False
    # P has no drafter: no rank line there (the launcher line names the form)
    assert "at boot registration" not in caplog.text


# ------------------------------------------- the admission funnel (Nadel path)

N_SLOTS = 64
FOREIGN = 7.0


class FakeDraftPool:
    """The MTP draft KV (1 layer); every row starts as a previous occupant's."""

    def __init__(self):
        self.layer_num = 1
        self.start_layer = 0
        self._k = torch.full((N_SLOTS, 4), FOREIGN)
        self._v = torch.full((N_SLOTS, 4), FOREIGN)

    def get_key_buffer(self, layer_id):
        return self._k

    def get_value_buffer(self, layer_id):
        return self._v


def admission_scheduler(ctl, pool):
    req_to_token = torch.arange(N_SLOTS, dtype=torch.int64).reshape(4, N_SLOTS // 4)
    ctl.draft_cold_spans = getattr(ctl, "draft_cold_spans", {})
    return types.SimpleNamespace(
        draft_worker=types.SimpleNamespace(
            draft_worker=types.SimpleNamespace(
                draft_runner=types.SimpleNamespace(token_to_kv_pool=pool))),
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        tree_cache=types.SimpleNamespace(cache_controller=ctl),
    )


def req(rid, idx, n_prefix, host_hit):
    return types.SimpleNamespace(rid=rid, req_pool_idx=idx,
                                 prefix_indices=list(range(n_prefix)),
                                 host_hit_length=host_hit)


def test_nf_restored_prefix_under_tier_off_is_zero_and_cold(monkeypatch, caplog):
    """THE NADEL PATH. x158 weg2-0-4: prefix 97840 = 1528 full pages (the
    store claim, #993 zero fill, KEPT) + a 48-row tail the claim does not
    cover. Scaled: page 4, prefix 11 = 2 pages + 3 tail rows, all restored.
    Tier off: no draft row comes back, so the funnel must scrub the WHOLE
    prefix and mark it cold -- the claimed span then holds exactly the zeros
    the armed #993 fill held (same drafter input), the tail zeros instead of
    a previous occupant's bytes. The extend rows stay (the batch writes them)."""
    form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    ctl = register()
    pool = FakeDraftPool()
    r = req("weg2-0-4", 1, 11, host_hit=11)
    with caplog.at_level(logging.INFO):
        out = arm_draft_cold_for_admission(admission_scheduler(ctl, pool), types.SimpleNamespace(reqs=[r]))
    rows = torch.arange(16, 32)
    assert torch.all(pool._k[rows[:11]] == 0) and torch.all(pool._v[rows[:11]] == 0)
    assert torch.all(pool._k[rows[11:]] == FOREIGN)
    assert out["cold"] == 1 and out["kept"] == 0 and rounds_owed(r) == 1
    msgs = " ".join(m.getMessage() for m in caplog.records)
    assert "draft tier armed=False" in msgs
    assert "span=[0,11) of 11" in msgs


def test_nf_device_radix_hit_stays_warm_under_tier_off(monkeypatch):
    """REGRESSION GUARD. x158 weg2-0-3: 'WEG2 DRAFT-WARM rid=weg2-0-3
    pages=3840', accept 2.60 -- a DEVICE radix hit inside one awake period,
    its draft rows written by D's own drafter. 'No producer on P' says
    nothing about those rows; turning the gate off must not make
    draft_cold_reason scrub them (#861's over-approximation was for rows a
    phase switch had reallocated). host_hit_length=0 = nothing restored."""
    form(monkeypatch, group="D", draft_on_p=False, tier="launcher")
    ctl = register()
    pool = FakeDraftPool()
    r = req("weg2-0-3", 2, 12, host_hit=0)
    out = arm_draft_cold_for_admission(admission_scheduler(ctl, pool), types.SimpleNamespace(reqs=[r]))
    assert torch.all(pool._k[32:48] == FOREIGN), "a device hit's own draft rows were scrubbed"
    assert out["cold"] == 0 and rounds_owed(r) == 0


# ------------------------------------------------------------- the host ledger

NF_MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
#: x158 geometry at the GROUP env (ARENA_GIB=4, 32 mamba slots): 349525 KV
#: slots x 12288 B, draft 349525 x 1024 B, mamba 32 x 58834944 B. The launcher
#: itself ran with ARENA_GIB unset (default 22): 1922389 draft slots = 1.83 GiB
#: of the ledger's arena=25.59 for a sparse arena-65536.bin nobody writes.
NF_KV_B, NF_DRAFT_B, NF_MAMBA_B = 349525 * 12288, 349525 * 1024, 32 * 58834944


def _nf_ledger(monkeypatch, tmp_path, arena_gib="4"):
    import os
    import shutil

    if not os.path.isfile(os.path.join(NF_MODEL, "config.json")):
        pytest.skip("Next-Flash config.json not on this host")
    shutil.copy(os.path.join(NF_MODEL, "config.json"), tmp_path / "config.json")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "32")
    if arena_gib is None:
        monkeypatch.delenv("SGLANG_HICACHE_ARENA_GIB", raising=False)
    else:
        monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", arena_gib)
    from sglang.srt.weg2 import launcher as L

    # the x158 form as main leaves it: NEXTN, H25 unresolved -> the env reads
    monkeypatch.setitem(L._SPEC_FORM, "form", "NEXTN")
    monkeypatch.setitem(L._SPEC_FORM, "draft_kv_on_p", None)
    return L._weg2_arena_ledger_terms(str(tmp_path))["arena_gib"] * (1 << 30)


def test_nf_ledger_prices_no_draft_arena_without_a_producer(monkeypatch, tmp_path):
    """RED before the pick: the launcher (no SGLANG_WEG2_GROUP of its own)
    with DRAFT_ON_P=False and the tier on auto still prices the draft arena."""
    form(monkeypatch, group=None, draft_on_p=False)
    got = _nf_ledger(monkeypatch, tmp_path)
    assert abs(got - (NF_KV_B + NF_MAMBA_B)) < 1 << 20, (got - NF_KV_B - NF_MAMBA_B) / (1 << 20)


def test_nf_ledger_keeps_the_draft_arena_with_the_tier_on(monkeypatch, tmp_path):
    form(monkeypatch, group=None, draft_on_p=False, tier="on")
    got = _nf_ledger(monkeypatch, tmp_path)
    assert abs(got - (NF_KV_B + NF_DRAFT_B + NF_MAMBA_B)) < 1 << 20


def test_nf_launcher_ledger_drops_the_1_83_gib_draft_term(monkeypatch, tmp_path, caplog):
    """x158's launcher (ARENA_GIB unset -> 22): 'WEG2-ARENA-LEDGER kv=1922389
    slots x 12288 B + draft 1922389 x 1024 B + ... = 25.59 GiB'. Under auto
    without a producer the draft term is 0 -- exactly 1922389 x 1024 B =
    1.83 GiB less -- and the ledger line says 'draft 0 x 1024 B'."""
    form(monkeypatch, group=None, draft_on_p=False)
    caplog.set_level(logging.INFO, logger="weg2.launcher")
    off = _nf_ledger(monkeypatch, tmp_path, arena_gib=None)
    monkeypatch.setenv(TIER_ENV, "on")
    on = _nf_ledger(monkeypatch, tmp_path, arena_gib=None)
    assert on - off == 1922389 * 1024
    assert round((on - off) / (1 << 30), 2) == 1.83
    assert "draft 0 x 1024 B" in caplog.text and "draft 1922389 x 1024 B" in caplog.text
