# htsglang README: sleep notice (text for the htsglang repository, NOT committed on main)

Plan row F0-I / A6 of `deskq/PLAN-RENAME-FLLIPER-1007.md`: when the renamed tree is pushed to `efschu/fLLiper` (main + tag `0.1.0`, by the
27B seat after the user's Go), the htsglang repository gets this block at the top of its `README.md` and goes to sleep. It is prepared here
on the integration branches only; nothing was committed on `main` of htsglang.

```markdown
> **This project has been renamed to fLLiper.** htsglang is dormant and receives no further changes. The renamed tree, the release
> 0.1.0 and its Duo image live in [efschu/fLLiper](https://github.com/efschu/fLLiper) (image `ghcr.io/efschu/flliper:0.1.0-cu130`).
> The published tag `ghcr.io/efschu/htsglang:cu130-nccl2307` stays as it is. Old names keep working in the new project
> (environment variables, flags, log markers, endpoints, cache directory; see its README, section "Coming from htsglang").
```

Apply (27B seat, after the push of fLLiper): put the block under the title line of the htsglang `README.md`, commit as efschu without
trailers, push to the htsglang fork. Until the user gives the Go for the fLLiper push the block would point at a repository that does not
exist: do not apply it earlier.
