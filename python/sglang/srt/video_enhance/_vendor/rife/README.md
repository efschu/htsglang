# Vendored RIFE IFNet architectures

## Origin

| Field | Value |
|---|---|
| Upstream repository | https://github.com/HolyWu/vs-rife |
| Version | `5.7.0` (`vsrife.__version__`) |
| Commit | `3488617283db7c428a83ba4a19382285da698b6a` (2026-05-10) |
| Licence | MIT, see `LICENSE` in this directory |
| Copyright holder | Copyright (c) 2021 HolyWu |
| Vendored on | 2026-07-31 |

The IFNet weights themselves are *not* vendored. They are fetched at runtime
from the upstream release
`https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v<VER>.pkl`
by `sglang.srt.video_enhance.rife.download_weights`, which records the sha256,
the source URL and the fetch time in a sidecar JSON.

## Files taken

Each file is a byte-for-byte copy of the upstream file at the commit above.
No edits of any kind were made, so `diff` against upstream stays empty and a
future upstream bump is a plain re-copy.

| File | upstream path | sha256 (upstream == vendored) |
|---|---|---|
| `warplayer.py` | `vsrife/warplayer.py` | `f17b5aea73676c059b4c155852217c954d5b4eb19cc08eff40b43edd6055ab73` |
| `IFNet_HDv3_v4_6.py` | `vsrife/IFNet_HDv3_v4_6.py` | `c45aed7bf7e2d2e8b8ba733fb4ad3ef4333df301ef0dba8300da34f861679092` |
| `IFNet_HDv3_v4_18.py` | `vsrife/IFNet_HDv3_v4_18.py` | `96816a3be79eeb1bb7de89bb27c92d2cb2ee5805357e721d8b230f4cc20a2647` |
| `IFNet_HDv3_v4_26.py` | `vsrife/IFNet_HDv3_v4_26.py` | `822e04b96fcb121a742eeedc848488279eb93ae7658d958c8275293bf6358b60` |

`LICENSE` is a verbatim copy of the upstream repository's `LICENSE`.

## What was changed

Nothing. The four Python files import only `torch`, `torch.nn`,
`torch.nn.functional` and the sibling `warplayer`; none of them touches
VapourSynth, so no import fix or de-coupling edit was required. The relative
import `from .warplayer import warp` resolves inside this package unchanged.

The VapourSynth coupling in upstream lives entirely in `vsrife/__init__.py`
(clip handling, frame callbacks, `vs.Error`). That file is deliberately **not**
vendored; its non-VapourSynth logic — the version enum, the modulo padding
rule, the `scale` semantics, the `module.`/`encode.` state-dict key rewriting —
is re-implemented in `sglang/srt/video_enhance/rife.py`. That split is the
DESIGN #333 §9.3 verdict for this component: "dependency at the model level,
port at the execution level".

## Why only three versions

Upstream ships 36 IFNet variants (4.0 through 4.26 with `.lite`/`.heavy`).
Vendoring all of them would mean carrying 36 near-duplicate files whose only
validation would be that they import. Three are vendored:

* `4.6` — the widely exercised baseline; it is one of the four versions
  VSGAN-tensorrt-docker actually benchmarks (`ANALYSE_333_prior_art_vsgan.md`
  §6.2). No `Head`/encode sub-network, modulo 32.
* `4.18` — the newest of the versions VSGAN benchmarks. Has `Head` with
  8 encode channels, modulo 32.
* `4.26` — current head of the enum. Has `Head` with 4 encode channels, five
  IFBlocks instead of four, and modulo 64 — so the padding rule and the
  block-count variation are both covered by something that is actually loaded.

`rife.py` keeps the full 36-entry enum as *known* versions and rejects a known
but non-vendored version by name. It never substitutes a different version.

## Re-vendoring procedure

```
git clone https://github.com/HolyWu/vs-rife /tmp/vs-rife
cd /tmp/vs-rife && git checkout <new-commit>
cp vsrife/warplayer.py vsrife/IFNet_HDv3_v4_{6,18,26}.py vsrife/LICENSE \
   python/sglang/srt/video_enhance/_vendor/rife/
```

Then update the table above (commit, version, sha256s) and re-run
`test/registered/video_enhance/test_rife.py`, which instantiates every vendored
IFNet and runs a CPU forward pass. Upstream has changed IFNet forward
signatures between versions before (4.26 returns a third `feat` tensor from
`IFBlock.forward` that 4.6 and 4.18 do not), so a bump is not assumed safe.
