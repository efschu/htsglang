# fLLiper

fLLiper is an LLM inference server for mismatched GPUs, derived from [SGLang](https://github.com/sgl-project/sglang)
(Apache-2.0, attribution in `LICENSE` and in the file headers). It was developed under the name **htsglang**; the
project is called fLLiper from release 0.1.0 on. The Python package is `flliper`, the runtime environment variables are
`FLLIPER_*`, the P/D-flip subsystem is `pdflip` (`FLLIPER_PDFLIP_*`, `--pdflip-*`, log markers `PDFLIP-*`).

> **Status.** Release 0.1.0 is prepared in the repository. The image is **not published** yet; pushing the tree to
> `github.com/efschu/fLLiper` and the image to `ghcr.io` are user gates. Everything below describes the state of the
> integration branches `desk/flliper-27b-int-1008` (27B line) and `desk/flliper-nf-int-1008` (Flash-Next line).

## What is in it

* **Heterogeneous tensor parallelism.** Ranks on chosen physical GPUs (`--rank-gpu-id`, resolved by NVML UUID, duplicates
  put several ranks on one card), a per-rank memory budget in MiB (`--rank-gpu-memory-mib`), uneven weight and KV shares
  (`--rank-tp-ratio`, `--rank-kv-ratio`, `--rank-mlp-ratio`, `--rank-moe-ratio`). `FEATURES_VS_UPSTREAM.md` lists every
  feature against upstream with the evidence label of each row.
* **pdflip: prefill/decode flipping on one box.** Two complete layouts of one model on the same cards, a pipeline-parallel
  group P for long prefill and a tensor-parallel group D for decode; the cards flip between them, so only one layout owns the
  GPUs at any moment. A request whose uncached part is at most X tokens is prefilled on D without a flip. The `27b-nvfp4-dual`
  profile is the experimental variant in which P and D are awake at the same time (no flip). Manual: `docker/pdflip-release/README_RELEASE_DRAFT.md` (in the 27B-line tree).
* **Profile planner.** From a hardware profile (NVML, persisted) and a model profile it proposes a launch profile for the
  operating forms single card, TP only, flip PP/TP and dual; every value can be overridden and carries its state
  (proposed / solved by the launcher / overridden by you / unproven), its executability (works / needs `--force` / refused with
  code) and its dependencies from the edge catalog (`python/flliper/srt/pdflip/kantenkatalog_1004.json`). The planner calls the
  launcher's own dry run as the oracle and does not change the launcher's arithmetic.
* **Dashboard** (`tools/rig_dashboard`, package `rigdash`). Live tiles, boots, history, cards, flip time, the profile editor
  and planner on one page, issue-text export of a hardware profile. Time series go to VictoriaMetrics and Grafana; the
  metric names (`sglang:*`, `weg2_*`) are deliberately unchanged so the history stays readable (RENAME_PLAN 8.17).

## Image (Duo, 0.1.0)

One image carries both model lines as two code states (`27b*` profiles run the 27B tree, `nf*` profiles the Flash-Next tree).
Tag scheme (from `docker/flliper/make_flat_ctx.sh`, F0-G): `ghcr.io/efschu/flliper:0.1.0-cu130` and
`flliper:cu130-<sha10>`; labels `org.opencontainers.image.*` and `io.github.efschu.flliper.*` (source
`https://github.com/efschu/fLLiper`, release `0.1.0`, `.revision`, `.revision.27b`, `.revision.nf`). The publish gate is
`docker/flliper/host_publish_flliper.sh`. The previous public tag `htsglang:cu130-nccl2307` stays untouched.

```bash
docker run -d --name flliper-27b --gpus all --security-opt apparmor=unconfined \
  --shm-size=48g --ulimit memlock=-1:-1 --init -p 127.0.0.1:30030:30030 \
  -e MODE=pdflip -e HTSGLANG_PROFILE=27b \
  ghcr.io/efschu/flliper:0.1.0-cu130 serve        # complete flags: README_RELEASE_DRAFT.md of the 27B-line tree, section 3.3
curl -s http://127.0.0.1:30030/pdflip/state          # readiness: {"state": "serving", ...}
```

The container's product environment keeps its old prefix (`HTSGLANG_*`, e.g. `HTSGLANG_PROFILE`) and the image-internal
paths `/opt/htsglang`, `/var/lib/htsglang` are unchanged in 0.1.0 (product layer, RENAME_PLAN phase 2b, see
`docs/dev/RENAME_PLAN.md` 8.19).

## Coming from htsglang (old names keep working)

| Old | New | Compatibility |
|---|---|---|
| `import sglang`, `python -m sglang.launch_server` | `flliper`, `python -m flliper.launch_server` | **no import alias in 0.1.0** (the meta-path alias of plan row F0-C is not built, `docs/dev/RENAME_PLAN.md` 8.19); user scripts change the import |
| `SGLANG_*`, `SGLANG_WEG2_*` | `FLLIPER_*`, `FLLIPER_PDFLIP_*` | the new name wins; a difference warns; deprecation line in the log |
| `--weg2-x` | `--pdflip-x` | the launcher parser accepts both |
| log markers `WEG2-*` | `PDFLIP-*` | tools of this repository read both |
| `/weg2/...` endpoints | `/pdflip/...` | both answer |
| `~/.cache/sglang` | `~/.cache/flliper` | read fallback |
| profile `X.env` (old names) | `X.env` (new names) | old files stay as `X.env.alt` |

Live profiles are converted with `tools/release/profconv.py --convert-live --apply` (operator, at the switch-over; the
gpu-arb patch `f0b-gpu-arb-1007-v2` first). Rename rules, must-keep list and the proofs: `docs/dev/RENAME_PLAN.md`.

## Release notes

`docs/dev/RELEASE_NOTES_FLLIPER_0.1.0.md`: scope, verification numbers (from the work-package reports), the fixes after
the freeze, open points.

## License

Apache-2.0, as upstream. Attribution lines, `LICENSE` and `NOTICE` are untouched by the rename.
