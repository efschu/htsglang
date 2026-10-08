# docker/flliper: build files and profiles of the Duo image (F0-G)

One image, two lines (`27b`, `nf`), built from the two renamed trees. The profile family picks the code stand
(`27b*` -> `/opt/htsglang/src-27b`, `nf*` -> `/opt/htsglang/src-nf`); the entrypoint is the same script for both.

| path | what |
|---|---|
| `Dockerfile.flliper` | flat build, labels `org.opencontainers.image.*` and `io.github.efschu.flliper.*` (source `https://github.com/efschu/fLLiper`, release `0.1.0`, `.revision`, `.revision.27b`, `.revision.nf`) |
| `make_flat_ctx.sh` | context generator. `--plan` read-only, `--check` = offline `--plan` where every warning is a failure, `--write` creates `ctx/flat-<release>-<sha10>-<cu>/` (builds nothing). Tags `flliper:0.1.0-cu130`, `flliper:cu130-<sha10>`. |
| `host_publish_flliper.sh`, `host_publish_flliper_selftest.sh` | publish gate (six locks) and its self-test |
| `profiles_release/*.env`, `profiles/nf-int4-h6-abl.env` | the release profiles in the NEW spelling (`FLLIPER_*`, `FLLIPER_PDFLIP_*`, `--pdflip-*`, `PDFLIP-*`); `*.env.alt` = the old file for comparison and fall-back: byte for byte except the sibling-`source` lines (10 of 16 profiles source a sibling), which name `<sibling>.env.alt`, so a `.alt` run is a pure old-spelling run (`profconv.py`: `alt_text()` / `from_alt()` are the exact inverse pair) |
| `drycmp/` | the old-vs-new dry-run comparison: `run_pair.sh` (one profile twice), `drycmp.py` (the comparison), `protocols/<profile>.txt` (one protocol per profile) and `protocols/SUMMARY.txt` (see below) |

`HTSGLANG_*` (the product environment of the container), evidence/host names (`boot_weg2_*`, `/spinning/gpu-arb/weg2`) and the
measured-data file formats (`weg2-x-curves/1`) are not renamed (RENAME_PLAN R2, F0-D).

## Regenerate the profile set from the live rig directories

    python3 tools/release/profconv.py --tree-out docker/flliper            # write
    python3 tools/release/profconv.py --tree-out docker/flliper --check    # exit 1 on a difference

## Build a context with the tree's own files

    bash docker/flliper/make_flat_ctx.sh --check --rev <27b-sha> --branch <27b-branch> --rev-nf <nf-sha> --branch-nf <nf-branch> \
         --profiles-from-tree <checkout>

`--profiles-from-tree` reads the profiles from `<checkout>/docker/flliper/`; without it the live rig dirs are read and a profile that
still carries a pre-rename name in a renamed tree is blocker B6.

## Dry-run comparison, old vs new (acceptance "0 diff modulo names")

    bash docker/flliper/drycmp/run_pair.sh <profile> <pre-rename tree> <this tree> <logdir>
    python3 docker/flliper/drycmp/drycmp.py --dir <logdir> --out docker/flliper/drycmp/protocols --old-tree <pre-rename tree> --new-tree <this tree> <profile>...

Old side = the pre-rename tree + `<profile>.env.alt`, new side = this tree + `<profile>.env`. A dry-run on this box ends where the launcher
refuses because model/draft files are invisible (`config.json`, W61, W128, W163) or where the plan itself is infeasible (W64); everything
computed before that line is compared, and `protocols/SUMMARY.txt` names the boundary per profile. That boundary is a limit of the check,
not a success: nothing behind it was compared.

## Convert the LIVE profile directories (operator, at the switch-over; an agent never does this)

    python3 tools/release/profconv.py --list-live                  # the files (profiles_release/*.env, profiles/*.env)
    python3 tools/release/profconv.py --convert-live               # plan: which files would change
    python3 tools/release/profconv.py --convert-live --apply       # X.env -> X.env.alt (kept), new X.env; idempotent
    python3 tools/release/profconv.py --rollback-live --apply      # the way back: X.env := the old file (from X.env.alt); plan without --apply

Apply the gpu-arb patch `f0b-gpu-arb-1007-v2` FIRST: `release_profile_gate.py` of the unpatched rig is red on converted profiles
(27B-L15, DUAL-MPS look for the old names).
