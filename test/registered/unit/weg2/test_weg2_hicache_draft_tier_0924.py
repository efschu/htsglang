"""HICACHE-DRAFT-TIER (user order 2026-09-24 14:15Z, verbatim: "und schreiben
wir in D auch draft context in den hicache? das muesste raus, weil draft ja
keinen hicacheplatz mehr bekommt").

Measured reason (boot weg2xsn420, --dflash-produce-on-p off): the draft arena
(720896 slots x 10240 B = 7.76 GiB) was pre-pinned on P-PP2 and every D rank
and carried nothing -- 30/30 D requests draft_pages=0, '#993 draft L3 READ'
>= 276728 pages asked, 0 hits. Hermetic, CPU. Pins:

Launcher (NF pick: the detection reads H25's resolved SGLANG_WEG2_DRAFT_ON_P /
--draft-kv-on-p -- the NF line has no --dflash-produce-on-p -- and these
launcher cases are written in that form; the rank cases are the 27B ones)
  * ONE detection of "group P produces draft pages", off P's FORM
    (p_group_has_draft_producer); ``auto`` (the default) resolves off without
    a producer, on with one; ``on``/``off`` force; anything else refuses (W151);
  * the RESOLVED ``off`` reaches BOTH groups; a resolved ``on`` writes nothing
    (the producer form's environment is byte-identical), spec_form_env is
    untouched;
  * the ledger prices no draft arena under ``off``; one launcher line.
Rank
  * ``off``: no draft host pool is registered (boot and cutover route), so
    has_draft stays False and the one gate answers False for every direction;
    no draft arena file is created; unset/auto/on register as before;
  * admission: a pure DEVICE prefix hit stays warm, a prefix that came back
    through the host tier is cold by name (#993 zeros + 1 bootstrap round);
  * the admission scrub never indexes a slot-mapped (DFlash window) draft pool
    with target slots; a mirror pool is scrubbed as before.
"""

import inspect
import logging
import os
import shutil
import types
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.weg2.launcher as L
from sglang.srt.mem_cache import hicache_storage as hs

ENV = "SGLANG_WEG2_HICACHE_DRAFT_TIER"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("SGLANG_WEG2_DRAFT_ON_P", raising=False)
    saved = dict(L._SPEC_FORM)
    try:
        yield
    finally:
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(saved)


def _form(tmp_path, form="NEXTN", draft_on_p=False, cli="on", produce="on"):
    """P's form as main leaves it: apply_spec_form, then H25's resolved
    draft_on_p (and HiCache enabled) installed beside resolve_draft_on_p.
    ``cli`` is --draft-kv-on-p's value, which H25 overrules (default on).
    ``produce`` is --dflash-produce-on-p (UNIFY S6, only read under DFLASH):
    ``on`` keeps this file's NF-era meaning "a DFLASH producer on P computes";
    ``off`` is the 27B standard form p_draft=cold (built, never asked)."""
    L.apply_spec_form(SimpleNamespace(
        spec_form=form, dflash_draft_path=str(tmp_path), dflash_block=8, dflash_window=2048,
        draft_kv_on_p=cli, dflash_produce_on_p=produce))
    L._SPEC_FORM["draft_kv_on_p"] = bool(draft_on_p)


# ------------------------------------------------------------------ launcher


def test_one_name_on_both_sides():
    assert L.HICACHE_DRAFT_TIER_ENV == hs.HICACHE_DRAFT_TIER_ENV == ENV
    assert L.HICACHE_DRAFT_TIER_DEFAULT == "auto"


def test_the_producer_is_read_off_ps_form(tmp_path, monkeypatch):
    cases = [
        (("NEXTN", False), False),   # H25 standard form (Nutzer-Order 24.09. 08:25Z)
        (("NEXTN", True), True),     # the H25 A/B arm: P carries the MTP producer
        (("DFLASH", False), False),
        (("DFLASH", True), True),    # --dflash-produce-on-p on: it computes
    ]
    for (form, dop), want in cases:
        _form(tmp_path, form, dop)
        assert L.p_group_has_draft_producer() is want, (form, dop)
    # UNIFY S6: p_draft=cold (27B 2026-09-24) -- the DFlash producer is built
    # on P but never asked, so P produces no draft pages.
    _form(tmp_path, "DFLASH", True, produce="off")
    assert L.p_group_has_draft_producer() is False
    _form(tmp_path, "NEXTN", True, produce="off")  # only read under DFLASH
    assert L.p_group_has_draft_producer() is True
    # Unresolved (desk, before main): the H25 env, never the CLI default "on".
    L.apply_spec_form(SimpleNamespace(spec_form="NEXTN", dflash_draft_path=str(tmp_path),
                                      dflash_block=8, dflash_window=2048, draft_kv_on_p="on"))
    assert L._SPEC_FORM["draft_kv_on_p"] is None
    assert L.p_group_has_draft_producer() is False, "H25 default 0 overrules --draft-kv-on-p on"
    monkeypatch.setenv("SGLANG_WEG2_DRAFT_ON_P", "1")
    assert L.p_group_has_draft_producer() is True


def test_auto_resolves_off_without_a_producer_and_on_with_one(tmp_path):
    _form(tmp_path)
    assert L.resolve_hicache_draft_tier() == ("off", "auto: group P has no draft producer")
    assert L.hicache_draft_tier_env() == {ENV: "off"}
    _form(tmp_path, draft_on_p=True)
    assert L.resolve_hicache_draft_tier()[0] == "on"
    assert L.hicache_draft_tier_env() == {}, "the producer form's environment is untouched"


def test_the_operator_forces_and_nonsense_refuses(tmp_path, monkeypatch):
    _form(tmp_path, draft_on_p=False, cli="off")
    monkeypatch.setenv(ENV, "off")
    assert L.resolve_hicache_draft_tier() == ("off", f"{ENV}=off (operator)")
    assert L.hicache_draft_tier_env() == {ENV: "off"}
    _form(tmp_path, draft_on_p=False)
    monkeypatch.setenv(ENV, "ON")
    assert L.resolve_hicache_draft_tier() == ("on", f"{ENV}=on (operator)")
    assert L.hicache_draft_tier_env() == {}
    monkeypatch.setenv(ENV, "maybe")
    with pytest.raises(L.Weg2LaunchRefused, match="W151 Weg2HicacheDraftTierInvalid"):
        L.resolve_hicache_draft_tier()


@pytest.mark.parametrize("form", ["DFLASH", "NEXTN"])
def test_off_against_a_computing_producer_refuses_by_name(tmp_path, monkeypatch, form):
    # the DFlash producer's first publish would raise (no registered draft
    # pool); the NEXTN one would compute for nothing
    _form(tmp_path, form=form, draft_on_p=True)
    monkeypatch.setenv(ENV, "off")
    with pytest.raises(L.Weg2LaunchRefused, match="W152 Weg2HicacheDraftTierConflict"):
        L.resolve_hicache_draft_tier()


def test_spec_form_env_is_byte_identical_in_both_forms(tmp_path, monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DFLASH_PLACEMENT", raising=False)
    for dop in (False, True):
        _form(tmp_path, form="DFLASH", draft_on_p=dop)
        # UNIFY S6: P's --dflash-produce-on-p switch rides P's env under DFLASH
        # (always written), independent of draft_on_p.
        assert L.spec_form_env("P") == {"SGLANG_WEG2_DFLASH_PRODUCE": "1"}
        assert L.spec_form_env("D") == {"SGLANG_DFLASH_WINDOW_POOL": "1"}
        _form(tmp_path, form="NEXTN", draft_on_p=dop)
        assert L.spec_form_env("P") == {} and L.spec_form_env("D") == {}


def test_build_env_hands_the_resolved_value_to_both_groups():
    src = inspect.getsource(L.build_env)
    assert "env.pop(HICACHE_DRAFT_TIER_ENV, None)" in src
    assert "env.update(hicache_draft_tier_env())" in src
    assert src.index("env.pop(HICACHE_DRAFT_TIER_ENV, None)") < src.index(
        "env.update(hicache_draft_tier_env())")
    # build_env has no group branch around it: P and D get the same value
    assert "group" not in src.split("env.pop(HICACHE_DRAFT_TIER_ENV, None)")[1].split("return env")[0]


def test_main_resolves_the_producer_and_names_the_tier():
    src = inspect.getsource(L.main)
    assert '_SPEC_FORM["draft_kv_on_p"] = bool(draft_kv_on_p)' in src
    # NF: after H25's resolve_draft_on_p, whose value draft_kv_on_p carries
    assert src.index("resolve_draft_on_p(") < src.index(
        '_SPEC_FORM["draft_kv_on_p"] = bool(draft_kv_on_p)')
    assert src.index('_SPEC_FORM["draft_kv_on_p"] = bool(draft_kv_on_p)') < src.index(
        "log(hicache_draft_tier_line())")


def test_the_line(tmp_path):
    _form(tmp_path)
    line = L.hicache_draft_tier_line()
    assert line.startswith("WEG2 HICACHE-DRAFT-TIER: off (auto: group P has no draft producer)")
    assert "\n" not in line and "no draft arena" in line
    _form(tmp_path, draft_on_p=True)
    assert L.hicache_draft_tier_line().startswith(
        "WEG2 HICACHE-DRAFT-TIER: on (auto: group P produces draft pages)")


MODEL = L.MODEL_DEFAULT
DRAFT = L.DFLASH_DRAFT_PATH_DEFAULT


@pytest.mark.skipif(not (os.path.isdir(MODEL) and os.path.isdir(DRAFT)),
                    reason="27B checkpoint / DFlash2 draft not on this host")
def test_the_ledger_prices_no_draft_arena_when_off(monkeypatch, caplog):
    monkeypatch.delenv("SGLANG_HICACHE_ARENA_HOST", raising=False)
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", "22")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "112")
    caplog.set_level(logging.INFO, logger="weg2.launcher")
    L.apply_spec_form(SimpleNamespace(spec_form="DFLASH", dflash_draft_path=DRAFT, dflash_block=8,
                                      dflash_window=2048))
    L._SPEC_FORM["draft_kv_on_p"] = True
    on = L._weg2_arena_ledger_terms(MODEL)["arena_gib"]
    L._SPEC_FORM["draft_kv_on_p"] = False
    off = L._weg2_arena_ledger_terms(MODEL)["arena_gib"]
    # 720896 KV slots x the DFlash2 draft page (5 layers x K+V x 8 heads x 128 x fp8)
    assert on - off == pytest.approx(720896 * 10240 / (1 << 30), abs=1e-9) == 6.875
    assert "draft 720896 x 10240 B" in caplog.text and "draft 0 x 10240 B" in caplog.text


# ---------------------------------------------------------------------- rank


class _Algo:
    def is_none(self):
        return False

    def is_ngram(self):
        return False


class _Ctl:
    def __init__(self):
        self.mem_pool_host = SimpleNamespace(size=4096)
        self.registered = None
        self.disarmed = None
        self.solo_draft_shadow = False

    def set_draft_kv_pool(self, dev, host, **kw):
        self.registered = (dev, host, kw)

    def disarm_draft_kv_pool(self, reason):
        self.disarmed = reason


def _register(monkeypatch, *, pool, shadow=False):
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    built = []
    monkeypatch.setattr(kcb, "_build_draft_host_pool",
                        lambda **kw: built.append(kw) or SimpleNamespace(size=4096, layer_num=1))
    monkeypatch.setattr(kcb, "drafter_identity_hash", lambda sa: "d" * 16)
    monkeypatch.setattr(kcb, "draft_total_kv_heads_of", lambda dw, sa: 8)
    runner = SimpleNamespace(token_to_kv_pool=pool, is_draft_solo_shadow=shadow)
    worker = SimpleNamespace(draft_worker=SimpleNamespace(draft_runner=runner))
    ctl = _Ctl()
    tree = SimpleNamespace(cache_controller=ctl)
    kcb.maybe_register_hicache_draft(
        tree_cache=tree, draft_worker=worker, spec_algorithm=_Algo(),
        server_args=SimpleNamespace(enable_multi_layer_eagle=False, hicache_mem_layout="layer_first",
                                    hicache_storage_backend="file"),
        enable_hierarchical_cache=True, page_size=1)
    return ctl, built


class _MHAPool:
    size = 1024
    layer_num = 1


def test_off_registers_no_draft_host_pool(monkeypatch, caplog):
    monkeypatch.setenv(ENV, "off")
    caplog.set_level(logging.INFO)
    ctl, built = _register(monkeypatch, pool=_MHAPool())
    assert ctl.registered is None and built == []
    assert "WEG2 HICACHE-DRAFT-TIER off" in caplog.text


def test_off_leaves_a_solo_shadow_unmarked_like_every_other_rank(monkeypatch):
    monkeypatch.setenv(ENV, "off")
    ctl, _ = _register(monkeypatch, pool=None, shadow=True)
    assert ctl.solo_draft_shadow is False
    monkeypatch.delenv(ENV)
    ctl, _ = _register(monkeypatch, pool=None, shadow=True)
    assert ctl.solo_draft_shadow is True, "without the switch the shadow marker is as before"


@pytest.mark.parametrize("value", [None, "auto", "on"])
def test_unset_auto_and_on_register_as_before(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    ctl, built = _register(monkeypatch, pool=_MHAPool())
    assert ctl.registered is not None and len(built) == 1
    assert ctl.registered[2]["owner_phase"] is None


def test_off_disarms_the_cutover_route_without_allocating(monkeypatch):
    from sglang.srt.mem_cache import kv_cache_builder as kcb

    monkeypatch.setattr(kcb, "_build_draft_host_pool", lambda **kw: pytest.fail("must not allocate"))
    monkeypatch.setattr(kcb, "resolve_draft_registration", lambda s, p: pytest.fail("must not resolve"))
    monkeypatch.setenv(ENV, "off")
    ctl = _Ctl()
    sched = SimpleNamespace(tree_cache=SimpleNamespace(cache_controller=ctl))
    assert kcb.rebind_hicache_draft_for_phase(sched, "tp") is False
    assert ctl.registered is None and "WEG2 HICACHE-DRAFT-TIER off" in ctl.disarmed


def test_an_unregistered_draft_half_is_disarmed_in_every_direction():
    from sglang.srt.managers.cache_controller import HiCacheController

    c = object.__new__(HiCacheController)
    c.has_draft = False
    for direction in ("write", "load", "l3-load", "admission"):
        assert c.draft_tier_armed(direction) is False


needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")


class _Store:
    _arena_dir = hs.HiCacheFile._arena_dir
    _arena_for = hs.HiCacheFile._arena_for
    _draft_arena_refused = hs.HiCacheFile._draft_arena_refused

    def __init__(self):
        self._canonical_kv_extents = SimpleNamespace(total_bytes=128)
        self.canonical_draft_page = SimpleNamespace(total_bytes=64)
        self.canonical_mamba_blob = SimpleNamespace(total_bytes=256)


@needs_gcc
def test_off_creates_no_draft_arena_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", "0.0000001")   # the 1024-slot floor
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "3")
    monkeypatch.setenv(ENV, "off")
    st = _Store()
    assert st._arena_for(64) is None
    assert not os.path.exists(tmp_path / "arena" / "arena-64.bin")
    assert st._arena_for(128).slots == 1024 and st._arena_for(256).slots == 3


@needs_gcc
def test_without_the_switch_the_draft_arena_is_created_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path / "arena"))
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_GIB", "0.0000001")
    st = _Store()
    assert st._arena_for(64).slots == 1024, "the KV slot count, byte-identical"


def test_admission_device_hit_stays_warm_host_restore_goes_cold(monkeypatch):
    from sglang.srt.managers import phase_flip_draft_bootstrap as b

    monkeypatch.setattr(b, "prefix_len", lambda req: 100)
    sched = SimpleNamespace(tree_cache=None)
    device_hit = SimpleNamespace(rid="a", needs_host_load_back=lambda: False)
    restored = SimpleNamespace(rid="b", needs_host_load_back=lambda: True)
    monkeypatch.setenv(ENV, "off")
    assert b.draft_cold_reason(sched, device_hit, tier_armed=False) is None
    assert "disarmed" in b.draft_cold_reason(sched, restored, tier_armed=False)
    # unknown provenance is the safe direction
    assert "disarmed" in b.draft_cold_reason(sched, SimpleNamespace(rid="c"), tier_armed=False)
    # without the switch a disarmed tier over-approximates as before
    monkeypatch.delenv(ENV)
    assert "disarmed" in b.draft_cold_reason(sched, device_hit, tier_armed=False)


class _Buffers:
    def __init__(self, rows):
        self.k = [torch.ones(rows, 2)]
        self.v = [torch.ones(rows, 2)]
        self.layer_num = 1
        self.start_layer = 0

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.v[layer_id]


def test_scrub_never_indexes_a_slot_mapped_pool_with_target_slots(monkeypatch):
    from sglang.srt.managers import phase_flip_draft_bootstrap as b

    monkeypatch.setattr(b, "draft_kv_layer_ids", lambda pool: [0])
    mapped = _Buffers(rows=4)
    mapped.weg2_slot_mapper = object()
    rows, layers = b.scrub_draft_kv(mapped, [torch.tensor([2, 900, 70000])])
    assert rows == 0 and layers == [0]
    assert bool(mapped.k[0].eq(1).all()), "a mapped pool's rows are never touched"
    mirror = _Buffers(rows=8)
    rows, _ = b.scrub_draft_kv(mirror, [torch.tensor([1, 3])])
    assert rows == 2 and mirror.k[0][1].sum() == 0 and mirror.k[0][0].sum() == 2
