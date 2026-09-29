# Vendored RIFE IFNet architectures

## Origin

| Field | Value |
|---|---|
| Upstream repository | https://github.com/HolyWu/vs-rife |
| Version | `5.7.0` (`vsrife.__version__`) |
| Commit | `3488617283db7c428a83ba4a19382285da698b6a` (2026-05-10) |
| Licence | MIT, see `LICENSE` in this directory |
| Copyright holder | Copyright (c) 2021 HolyWu |
| Vendored on | 2026-07-31, extended 2026-08-03 (#460) |

The IFNet weights themselves are *not* vendored. They are fetched at runtime
from the upstream release
`https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v<VER>.pkl`
by `sglang.srt.video_enhance.rife.download_weights`, which records the sha256,
the source URL and the fetch time in a sidecar JSON, and refuses a download
whose hash does not match `rife.KNOWN_WEIGHT_SHA256`. Use
`scripts/video_enhance/fetch_rife_weights.py` to obtain them; it will not
download an unpinned artifact twice.

## Files taken

Each file is the upstream file at the commit above, **reformatted by this
repository's `black` hook** and otherwise unedited. The reformat is line
wrapping only — no identifier, expression or statement differs, so the graphs
are the upstream graphs. It does mean `diff` against upstream is *not* empty
and the vendored sha256 is *not* the upstream one, so both are recorded below:
the upstream column is what a re-vendoring must fetch, the vendored column is
what must be on disk here after `black` has run.

| File | upstream path | upstream sha256 | vendored sha256 (post-`black`) |
|---|---|---|---|
| `warplayer.py` | `vsrife/warplayer.py` | `f17b5aea73676c059b4c155852217c954d5b4eb19cc08eff40b43edd6055ab73` | `5ca616cf9c235adc12799d84ab86d4b6e4b70b9674c68384359bf6164b92b165` |
| `IFNet_HDv3_v4_6.py` | `vsrife/IFNet_HDv3_v4_6.py` | `c45aed7bf7e2d2e8b8ba733fb4ad3ef4333df301ef0dba8300da34f861679092` | `2fef20510c17e343c2fd95f627f859999f15fd66b4250121cf95c8cbe95ffc30` |
| `IFNet_HDv3_v4_18.py` | `vsrife/IFNet_HDv3_v4_18.py` | `96816a3be79eeb1bb7de89bb27c92d2cb2ee5805357e721d8b230f4cc20a2647` | `459bbced1bf31d422fa1d00c42d621c4c534f8a359042def62e21afb5f5a8258` |
| `IFNet_HDv3_v4_17_lite.py` | `vsrife/IFNet_HDv3_v4_17_lite.py` | `57ec7e07b42eed83625f1da52da168f10dcdc08f8b7854af920b91ba88eab65d` | `4a57d522be927a9127e4bfa62da11bc77daedb74bef228a1d690d0f74661c9d4` |
| `IFNet_HDv3_v4_26.py` | `vsrife/IFNet_HDv3_v4_26.py` | `822e04b96fcb121a742eeedc848488279eb93ae7658d958c8275293bf6358b60` | `9aa551afc40c9349edff0660855e69033add4c29b395c2c84029479ca26ecdea` |

`LICENSE` is a verbatim copy of the upstream repository's `LICENSE`.

*(Correction, 2026-08-03: this table previously claimed the files were
byte-for-byte copies and listed only the upstream hashes, which do not match
the files in this directory. The claim was wrong from the first commit — the
`black` hook reformatted them on the way in. Nothing about the models changed;
only the documentation did.)*

## Why only four files for eight versions

Upstream ships 36 IFNet variants (4.0 through 4.26 with `.lite`/`.heavy`).
Several of those files are **byte-identical to each other**, so vendoring one
member of a group covers the whole group. Verified at the pinned commit:

| upstream files | upstream sha256 | versions covered here |
|---|---|---|
| `IFNet_HDv3_v4_15.py`, `IFNet_HDv3_v4_17.py`, `IFNet_HDv3_v4_18.py` | `96816a3b…` | `4.15`, `4.17`, `4.18` |
| `IFNet_HDv3_v4_15_lite.py`, `IFNet_HDv3_v4_16_lite.py`, `IFNet_HDv3_v4_17_lite.py` | `57ec7e07…` | `4.15.lite`, `4.16.lite`, `4.17.lite` |

`rife._VENDORED` maps the aliases onto the single vendored module. This is not
the substitution `require_supported` refuses: it is the same bytes under two
names. The **weights still differ per version** and are pinned separately in
`rife.KNOWN_WEIGHT_SHA256`, which is where the actual difference between 4.15
and 4.18 lives.

Reproduce the identity claim with:

```
sha256sum /tmp/vs-rife/vsrife/IFNet_HDv3_v4_{15,17,18}.py \
          /tmp/vs-rife/vsrife/IFNet_HDv3_v4_{15,16,17}_lite.py
```

The remaining two vendored architectures are genuinely distinct:

* `4.6` — the widely exercised baseline; it is one of the four versions
  VSGAN-tensorrt-docker actually benchmarks (`ANALYSE_333_prior_art_vsgan.md`
  §6.2). No `Head`/encode sub-network, modulo 32.
* `4.26` — current head of the enum. Has `Head` with 4 encode channels, five
  IFBlocks instead of four, and modulo 64 — so the padding rule and the
  block-count variation are both covered by something that is actually loaded.

The 4.15/4.17/4.18 group has `Head` with 8 encode channels and modulo 32; the
lite group has `Head` with 4 encode channels at half the IFBlock width and
modulo 32.

`rife.py` keeps the full 36-entry enum as *known* versions and rejects a known
but non-vendored version by name. It never substitutes a different version.

## Re-vendoring procedure

```
git clone https://github.com/HolyWu/vs-rife /tmp/vs-rife
cd /tmp/vs-rife && git checkout <new-commit>
cp vsrife/warplayer.py vsrife/IFNet_HDv3_v4_{6,18,26}.py \
   vsrife/IFNet_HDv3_v4_17_lite.py vsrife/LICENSE \
   python/sglang/srt/video_enhance/_vendor/rife/
black python/sglang/srt/video_enhance/_vendor/rife/
```

The `black` step is not optional — without it the repo's own pre-commit hook
rewrites the files anyway, and the vendored-hash column above would be a
record of a state that never existed on disk. Re-check the byte-identity
groups after a bump: upstream is free to make 4.17 diverge from 4.18 at any
release, and `_VENDORED` would then be mapping two different graphs onto one
module.

Then update both hash columns above and re-run
`test/registered/video_enhance/test_rife.py`, which instantiates every vendored
IFNet and runs a CPU forward pass. Upstream has changed IFNet forward
signatures between versions before (4.26 returns a third `feat` tensor from
`IFBlock.forward` that 4.6 and 4.18 do not), so a bump is not assumed safe.
