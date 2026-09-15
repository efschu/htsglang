"""#1378 xsn64 -- the boot nonce must be unique per launch on the ringless arm.

MEASURED on weg2xsn64 (9314c34d52): every P and D rank refused its manifest
write with W68 ("already holds a manifest of THIS boot with a DIFFERENT
inventory"), because the file on disk was weg2xsn63's -- a DIFFERENT boot --
and both carried ``boot_token=0``. `HostRingPlan.epoch` defaulted to 0 and the
#1369 ABSENT-BY-DESIGN arm of `prepare_host_ring` (every boot since
2026-09-14) never set it; `str(plan.epoch)` is the nonce the launcher
publishes as SGLANG_WEG2_XCHG_BOOT. Every earlier boot passed the ratchet
only because the inventory never changed between boots.

Properties:
  * a fresh HostRingPlan carries a non-zero epoch (mutant: the shipped
    default 0);
  * a stale manifest of ANOTHER token is overwritten silently by this boot's
    write, whatever its inventory (this is what a unique nonce buys);
  * the W68 ratchet still fires for the SAME token with a different
    inventory (control -- the ratchet is not disarmed, it is scoped).

Hermetic: CUDA_VISIBLE_DEVICES="", a tmp dir, no torch.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402


def _piece(name, rows, cols, *, tag="weights_0"):
    return xm.ManifestPiece(
        param_name=name,
        tensor_class=sh.tensor_class(name),
        rows_full=rows, cols_full=cols, itemsize=2,
        tag=tag, nbytes=rows * cols * 2,
        component_rows=(),
    )


def _manifest(token, names):
    return xm.RankManifest(
        group="P", rank=0, card=0, region_tag="weights_0",
        boot_token=str(token),
        pieces=tuple(_piece(n, 8, 8) for n in names),
        tp_rank=0, pp_rank=0,
    )


_WITH_TOWER = ("model.layers.0.input_layernorm.weight",
               "visual.patch_embed.proj.weight")
_TEXT_ONLY = ("model.layers.0.input_layernorm.weight",)


def test_fresh_plan_has_a_nonzero_epoch():
    from sglang.srt.weg2 import launcher

    plan = launcher.HostRingPlan()
    assert int(plan.epoch) > 0, plan.epoch
    # and it is the launch time, the same currency the front falls back to
    import time

    assert abs(int(plan.epoch) - int(time.time())) < 60


def test_stale_manifest_of_another_boot_is_overwritten(tmp_path):
    d = str(tmp_path)
    xm.write_rank_manifest(_manifest("0", _WITH_TOWER), d)      # xsn63's file
    path = xm.write_rank_manifest(_manifest("1789457229", _TEXT_ONLY), d)
    got = xm.RankManifest.from_json(json.load(open(path)), path=path)
    assert got.boot_token == "1789457229"
    assert tuple(p.param_name for p in got.pieces) == _TEXT_ONLY


def test_w68_still_fires_for_the_same_boot_token(tmp_path):
    d = str(tmp_path)
    xm.write_rank_manifest(_manifest("1789457229", _WITH_TOWER), d)
    with pytest.raises(Exception, match="W68"):
        xm.write_rank_manifest(_manifest("1789457229", _TEXT_ONLY), d)


def test_load_filters_the_stale_token(tmp_path):
    d = str(tmp_path)
    xm.write_rank_manifest(_manifest("0", _WITH_TOWER), d)
    assert xm.load_manifests(d, boot_token="1789457229") == ()
    xm.write_rank_manifest(_manifest("1789457229", _TEXT_ONLY), d)
    assert len(xm.load_manifests(d, boot_token="1789457229")) == 1
