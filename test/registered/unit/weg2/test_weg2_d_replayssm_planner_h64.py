# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H64 -- der D-Planer bucht unter --d-replayssm-spec den Ring, nicht die
Zwischenzustaende.

DER BEFUND: Das 27B-ReplaySSM-Paket (S5) bepreist den Ring in der Runtime (Mixin)
und im PerfCostModel. Der Planer, der auf der Next-Flash-Linie die D-Kante setzt
(``expert_residency``: #145 FRACTION-SOLVE gegen das Budget, KARTE D gegen die
Karte), rechnet aber mit GEMESSENEN Referenzen (fnFL2x151 + x158, x141 + x144),
und die liefen mit dem rekurrenten Verify: ``spec_mib`` 224.3 MiB auf D-TP0 und
ein Karten-Peak, der die Zwischenzustaende (2 Zeilen x 4 Draft-Schritte x
36 GDN-Layer x 48x128x128 bf16 = 432 MiB) traegt. Ohne Umbuchung bliebe der
Gewinn beim Planer liegen -- und D's KV ist dort auf 262144 Token gedeckelt, der
Gewinn geht also nur ueber die Experten-Zeilen-Kante an die Karte.

Gepinnt:

1. die Einheits-Bytes des Planers SIND die Laufzeitformeln
   (``mamba_cache_per_req``, ``spec_ring_workspace_bytes_per_req``,
   ``replayssm_ring_bytes_per_req``) und die Allokation eines echten
   ``MambaPool`` in beiden Formen;
2. die Umbuchung der eingebauten NF-Referenz: 224.3 -> 22.3 MiB Posten,
   431.7 -> 36.2 MiB Verify-Allokation auf rang0, 0 auf den Form-A-Workern;
3. die Form einer Referenz ist die WIRKUNG im Log ("GDN ReplaySSM SPEC ring
   allocated (L=..."), gemischte Referenzen werden verweigert;
4. am Launcher (Dry-Run der x166-Form): Schalter aus = byte-gleich (KANTE
   [133, 137, 134] = SCRATCH <= [121, 63, 49], wie H50), Schalter an = KANTE
   [136, 137, 134] = SCRATCH <= [124, 63, 49].
"""

import json
import os
import types
from unittest import mock

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.planner import expert_residency as er
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

MIB = float(1 << 20)
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_h39_h50")
MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
SLOT_BYTES = 1297637376 / 512
N_LAYERS = 48
DRY_BUDGETS = (29368, 18184, 17784)

#: Die GDN-Geometrie des Next-Flash-Checkpoints (config.json text_config):
#: 36 von 48 Layern sind linear_attention, 48 v- / 16 k-Koepfe, K = V = 128,
#: Conv-Kern 4, Modell-dtype bf16, Checkpoint-SSM-dtype float32.
NF_TEXT_CFG = {
    "num_hidden_layers": 48,
    "vocab_size": 248320,
    "hidden_size": 2560,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 48,
    "linear_num_key_heads": 16,
    "linear_conv_kernel_dim": 4,
    "dtype": "bfloat16",
    "mamba_ssm_dtype": "float32",
}
NF_GDN_LAYERS = 36
#: Die D-Form der Next-Flash-Bestform (boot_nf_bestform_0924.sh, x166-Log:
#: speculative_num_draft_tokens=4, max_running_requests=1, mamba_ssm_dtype
#: bfloat16) mit dem Ring der Laenge 16.
NF_RING = er.ReplaySSMSpecForm(
    ring_len=16, draft_tokens=4, max_running=1, ssm_dtype="bfloat16"
)


def _params(hv, hk, layers, *, ssm=torch.bfloat16, conv=torch.bfloat16, K=128, V=128, kc=4):
    conv_dim = 2 * hk * K + hv * V
    shape = Mamba2StateShape(
        conv=[(conv_dim, kc - 1)],
        temporal=(hv, V, K),
        intermediate_size=hv * V,
        conv_dim=conv_dim,
        ssm_state_size=K,
        num_heads=hv,
        head_dim=V,
        state_size=K,
        conv_kernel=kc,
        num_k_heads_per_tp=hk,
    )
    return Mamba2CacheParams(
        shape=shape,
        layers=list(range(layers)),
        dtype=Mamba2StateDType(conv=conv, temporal=ssm),
    )


# ---------------------------------------------------------------------------
# 1. die Einheits-Bytes sind die Laufzeitformeln
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ssm_name,ssm,act_name,act",
    [
        ("bfloat16", torch.bfloat16, "bfloat16", torch.bfloat16),
        ("float32", torch.float32, "bfloat16", torch.bfloat16),
        ("float32", torch.float32, "float32", torch.float32),
    ],
)
def test_unit_bytes_are_the_runtime_formulas(ssm_name, ssm, act_name, act):
    units, layers, D, L = 16, NF_GDN_LAYERS, 4, 16
    params = _params(48, 16, layers, ssm=ssm, conv=act)
    u = er.gdn_spec_unit_bytes(
        NF_TEXT_CFG, draft_tokens=D, ring_len=L, ssm_dtype=ssm_name, act_dtype=act_name
    )
    scale = units * layers
    assert u.per_req * scale == params.mamba_cache_per_req
    assert u.ring * scale == params.replayssm_ring_bytes_per_req(L)
    assert (u.conv_window + u.ring) * scale == params.spec_ring_workspace_bytes_per_req(D, L)
    ssm_state = 48 * 128 * 128 * torch.empty((), dtype=ssm).element_size()
    assert u.ssm * units == ssm_state
    # the recurrent verify has no ring
    assert er.gdn_spec_unit_bytes(
        NF_TEXT_CFG, draft_tokens=D, ring_len=None, ssm_dtype=ssm_name, act_dtype=act_name
    ).ring == 0.0


def test_both_forms_match_a_real_pool():
    """Posten -> per_req -> Allokation, gegen einen echten MambaPool (CPU) in
    BEIDEN Formen: rekurrent (intermediate_ssm) und Ring (d, k, g, Low-Parts)."""
    from sglang.srt.mem_cache import memory_pool as mp
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    layers, D, L, mrr = 2, 4, 16, 1
    params = _params(48, 16, layers)
    per_req_mib = params.mamba_cache_per_req / MIB
    form_ring = er.ReplaySSMSpecForm(ring_len=L, draft_tokens=D, max_running=mrr, ssm_dtype="bfloat16")
    form_rec = er.ReplaySSMSpecForm(ring_len=None, draft_tokens=D, max_running=mrr, ssm_dtype="bfloat16")
    got = {}
    for name, ring in (("rec", False), ("ring", True)):
        with mock.patch.object(mp, "conv_window_dedup_enabled", lambda *a: True):
            pool = mp.MambaPool(
                size=2,
                spec_state_size=mrr,
                cache_params=params,
                mamba_layer_ids=list(range(layers)),
                device="cpu",
                speculative_num_draft_tokens=D,
                speculative_eagle_topk=1,
                linear_replayssm_cache_len=L,
                enable_linear_replayssm_spec=ring,
            )
        c = pool.mamba_cache
        if ring:
            assert c.intermediate_ssm is None
            got[name] = sum(
                getattr(c, f).numel() * getattr(c, f).element_size()
                for f in ("replayssm_d", "replayssm_k", "replayssm_g", "replayssm_rawv", "replayssm_rawk")
            )
        else:
            got[name] = c.intermediate_ssm.numel() * c.intermediate_ssm.element_size()
    # a recurrent reference logs per_req x capped x D as its post
    rb = er.replayssm_spec_rebook(
        spec_mib_ref=(per_req_mib * mrr * D,),
        ref_ring_len=None,
        form=form_ring,
        text_cfg=NF_TEXT_CFG,
    )
    assert rb.alloc_ref_mib[0] == pytest.approx(got["rec"] / MIB, abs=0.05)
    assert rb.alloc_mib[0] == pytest.approx(got["ring"] / MIB, abs=0.05)
    assert rb.spec_mib[0] == pytest.approx(
        mrr * params.spec_ring_workspace_bytes_per_req(D, L) / MIB, abs=0.05
    )
    # and back: a ring reference re-booked for a recurrent boot
    back = er.replayssm_spec_rebook(
        spec_mib_ref=(mrr * params.spec_ring_workspace_bytes_per_req(D, L) / MIB,),
        ref_ring_len=L,
        form=form_rec,
        text_cfg=NF_TEXT_CFG,
    )
    assert back.spec_mib[0] == pytest.approx(per_req_mib * mrr * D, abs=0.05)
    assert back.alloc_mib[0] == pytest.approx(got["rec"] / MIB, abs=0.05)


# ---------------------------------------------------------------------------
# 2. die NF-Referenz, umgebucht
# ---------------------------------------------------------------------------


def test_nf_reference_rebook_numbers():
    ref = er.D_RESIDENCY_REFERENCE_FNFL2_H39
    assert ref.spec_mib == (224.3, 0.0, 0.0) and ref.replayssm_spec_ring_len is None
    rb = er.replayssm_spec_rebook(
        spec_mib_ref=ref.spec_mib, ref_ring_len=None, form=NF_RING, text_cfg=NF_TEXT_CFG
    )
    assert rb.spec_mib == (22.3, 0.0, 0.0)
    assert rb.alloc_ref_mib == (431.7, 0.0, 0.0)
    assert rb.alloc_mib == (36.2, 0.0, 0.0)
    assert rb.freed_mib == (395.5, 0.0, 0.0)
    # the exact TP0 allocation (all 16 GDN units on the Form-A host): the
    # reference's rounded post (0.219 GiB) costs < 0.5 MiB
    params = _params(48, 16, NF_GDN_LAYERS)
    rows, D = 2, 4
    exact_rec = rows * D * NF_GDN_LAYERS * 48 * 128 * 128 * 2 / MIB
    exact_ring = rows * params.replayssm_ring_bytes_per_req(16) / MIB
    assert exact_rec == 432.0
    assert rb.alloc_ref_mib[0] == pytest.approx(exact_rec, abs=0.5)
    assert rb.alloc_mib[0] == pytest.approx(exact_ring, abs=0.5)
    assert rb.spec_mib[0] == pytest.approx(
        params.spec_ring_workspace_bytes_per_req(D, 16) / MIB, abs=0.5
    )


def test_reference_form_is_read_from_the_log():
    line = (
        "[2026-09-24 20:18:49 TP0] GDN ReplaySSM SPEC ring allocated (L=16, request "
        "rows=2, intermediate_ssm skipped): d=0.013GB k=0.004GB g=0.000GB low parts=0.018GB"
    )
    assert er.boot_replayssm_spec_ring_len(line) == 16
    assert er.boot_replayssm_spec_ring_len("Mamba Cache is allocated.") is None

    def _t(n):
        with open(os.path.join(FIX, n + ".D.lines")) as fh:
            return fh.read()

    kw = dict(n_ranks=3, n_layers=N_LAYERS, slot_bytes=SLOT_BYTES, model=MODEL, rank_tp_ratio="1,0,0")
    rec = er.d_rank_reference_from_logs([("x151", _t("fnFL2x151")), ("x158", _t("fnFL2x158"))], **kw)
    assert rec.replayssm_spec_ring_len is None
    ring = er.d_rank_reference_from_logs(
        [("x151", _t("fnFL2x151") + "\n" + line), ("x158", _t("fnFL2x158") + "\n" + line)], **kw
    )
    assert ring.replayssm_spec_ring_len == 16
    with pytest.raises(ValueError, match="Verify-Form"):
        er.d_rank_reference_from_logs(
            [("x151", _t("fnFL2x151") + "\n" + line), ("x158", _t("fnFL2x158"))], **kw
        )
    card = er.d_card_reference_from_logs(
        [("x151", _t("fnFL2x151") + "\n" + line), ("x158", _t("fnFL2x158") + "\n" + line)], **kw
    )
    assert card.replayssm_spec_ring_len == 16
    # the shipped references are the recurrent boots they were measured on
    for r in er.D_RESIDENCY_REFERENCES + er.D_CARD_REFERENCES:
        assert r.replayssm_spec_ring_len is None


# ---------------------------------------------------------------------------
# 3. am Launcher: der Dry-Run der x166-Form, Schalter aus und an
# ---------------------------------------------------------------------------

NF_EXTRA_D = (
    "--max-running-requests 1 --mamba-ssm-dtype bfloat16 --rank-role host,worker,worker "
    "--rank-tp-ratio 1,0,0 --rank-moe-ratio 183,137,168 "
    "--rank-moe-resident-fraction 0.06,0.51,0.48 --speculative-draft-placement solo "
    "--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 4"
)


def _launcher_ns(tmp_path, scratch0):
    cfg = {"text_config": NF_TEXT_CFG}
    (tmp_path / MODEL).mkdir(parents=True)
    (tmp_path / MODEL / "config.json").write_text(json.dumps(cfg))
    env = (
        "SGLANG_MOE_POOL_STAGING=12;SGLANG_MOE_SCRATCH_SLOTS=%d,48,48;"
        "SGLANG_UNEVEN_MOE_EXPERT_SHARD=1;SGLANG_WEG2_DRAFT_SHARE_EMBED=1" % scratch0
    )
    return types.SimpleNamespace(
        model=str(tmp_path / MODEL),
        extra_d=NF_EXTRA_D,
        env_d=env,
        extra_p="",
        env_p="SGLANG_MOE_SCRATCH_SLOTS=32",
        pp_cut_expert_device_fraction="0.26,0.45,0.733887",
        pp_cut_expert_lru_rows="32",
        d_foreign_context_mib="1446,896,894",
        d_nontorch_mib="1981,528,524",
        d_reserve_mib="",
        d_residency_reference_logs="",
        d_card_reference_logs="",
        wake_credit_reference_logs="",
        d_bs=1,
        profile="nextflash",
    )


@pytest.fixture
def launcher_env(monkeypatch):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(
        pp_cut,
        "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(
            expert_layer_weight_bytes=1297637376.0, num_experts=512, n_layers=48
        ),
    )
    saved = dict(launcher._SPEC_FORM)
    try:
        yield launcher
    finally:
        launcher._SPEC_FORM.clear()
        launcher._SPEC_FORM.update(saved)


def _dry(launcher, tmp_path, name, scratch0, ring):
    from sglang.srt.environ import envs

    ns = _launcher_ns(tmp_path / name, scratch0)
    launcher.apply_spec_form(
        types.SimpleNamespace(
            spec_form="NEXTN", extra_d=ns.extra_d, d_replayssm_spec="on" if ring else "off"
        )
    )
    cards = [
        launcher.Card(1, "u1", "RTX 5090", 32607),
        launcher.Card(0, "u0", "RTX 3080", 20480),
        launcher.Card(2, "u2", "RTX 3080", 20480),
    ]
    lines = []
    with envs.SGLANG_WEG2_DRAFT_ON_P.override(False):
        launcher.log_d_rank_vram_solve(
            ns, cards, DRY_BUDGETS, lines.append, "D(dry, expectation)",
            p_split=[29, 11, 8], chunk_layers=3,
        )
    return ns, lines


def test_plan_form_is_none_when_off_and_reads_d_when_on(launcher_env, tmp_path):
    launcher = launcher_env
    ns = _launcher_ns(tmp_path / "f", 118)
    launcher.apply_spec_form(types.SimpleNamespace(spec_form="NEXTN", extra_d=ns.extra_d))
    assert launcher.d_replayssm_spec_plan_form(ns) is None
    launcher.apply_spec_form(
        types.SimpleNamespace(spec_form="NEXTN", extra_d=ns.extra_d, d_replayssm_spec="on")
    )
    assert launcher.d_replayssm_spec_plan_form(ns) == NF_RING
    # without --extra-d depth/mrr/dtype: the constants' window, --d-bs, the checkpoint's
    ns2 = types.SimpleNamespace(extra_d="", d_bs=2, profile="nextflash")
    launcher.apply_spec_form(types.SimpleNamespace(spec_form="NEXTN", extra_d="", d_replayssm_spec="on"))
    f = launcher.d_replayssm_spec_plan_form(ns2)
    assert (f.draft_tokens, f.max_running, f.ssm_dtype) == (int(launcher.SPEC_NUM_DRAFT_TOKENS), 2, None)


def test_dry_run_off_is_unchanged_on_moves_the_edge(launcher_env, tmp_path):
    launcher = launcher_env
    _, off = _dry(launcher, tmp_path, "off", 121, ring=False)
    head = [ln for ln in off if "KARTE D(dry, expectation) (H33" in ln]
    assert head and "Zeilen [133, 137, 134] = SCRATCH <= [121, 63, 49]" in head[0], off
    assert not any("REPLAYSSM-SPEC" in ln for ln in off)
    r0 = [ln for ln in off if "FRACTION-SOLVE D(dry, expectation) rang0" in ln][0]
    assert "spec 224 +" in r0

    _, on = _dry(launcher, tmp_path, "on", 121, ring=True)
    head = [ln for ln in on if "KARTE D(dry, expectation) (H33" in ln]
    assert head and "Zeilen [136, 137, 134] = SCRATCH <= [124, 63, 49]" in head[0], on
    rebook = [ln for ln in on if "FRACTION-SOLVE D(dry, expectation) REPLAYSSM-SPEC (H64)" in ln]
    assert rebook and "224.3 -> 22.3, 0.0 -> 0.0, 0.0 -> 0.0" in rebook[0], on
    assert "431.7 -> 36.2" in rebook[0] and "GERECHNET" in rebook[0]
    shift = [ln for ln in on if "KARTE D(dry, expectation) REPLAYSSM-SPEC (H64)" in ln]
    assert shift and "['+395.5', '+0.0', '+0.0']" in shift[0], on
    r0 = [ln for ln in on if "FRACTION-SOLVE D(dry, expectation) rang0" in ln][0]
    assert "spec 22 +" in r0
    # everything that is not the spec post / the TP0 card is the off plan
    tp12 = lambda ls: [ln for ln in ls if ("rang1:" in ln or "rang2:" in ln)]  # noqa: E731
    assert tp12(on) == tp12(off)


def test_dry_run_edge_scratch_under_the_ring(launcher_env, tmp_path):
    launcher = launcher_env
    # the new edge passes, one row past it dies at the card (W130) ...
    _dry(launcher, tmp_path, "e124", 124, ring=True)
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 Weg2DCardNearOom"):
        _dry(launcher, tmp_path, "e125", 125, ring=True)
    # ... and without the ring the same 124 is refused (the old edge is 121)
    with pytest.raises(launcher.Weg2LaunchRefused, match=r"^W130 Weg2DCardNearOom"):
        _dry(launcher, tmp_path, "e124off", 124, ring=False)


def test_plan_d_residency_without_the_form_is_byte_identical(tmp_path, monkeypatch):
    """Schalter aus: der Launcher uebergibt None -- dieselben Zeilen wie ohne den
    Parameter (der Aufruf vor H64)."""
    from sglang.srt.planner import pp_cut

    monkeypatch.setattr(
        pp_cut,
        "checkpoint_weight_terms",
        lambda _p: types.SimpleNamespace(
            expert_layer_weight_bytes=1297637376.0, num_experts=512, n_layers=48
        ),
    )
    ns = _launcher_ns(tmp_path, 118)
    kw = dict(
        model_path=ns.model,
        budgets_mib=DRY_BUDGETS,
        ratios=(183, 137, 168),
        fractions=(0.06, 0.51, 0.48),
        scratch_rows=(118, 48, 48),
        rank_tp_ratio="1,0,0",
        env_d=dict(kv.split("=", 1) for kv in ns.env_d.split(";")),
        reference_logs="",
        kv_tokens=262144,
        label="D(dry, expectation)",
        marker="D-RANK VRAM (#145)",
    )
    a = er.plan_d_residency(**kw)
    b = er.plan_d_residency(**kw, replayssm_spec=None)
    assert a.lines == b.lines and a.refusal == b.refusal
    assert a.fits == b.fits and a.card_fits == b.card_fits
