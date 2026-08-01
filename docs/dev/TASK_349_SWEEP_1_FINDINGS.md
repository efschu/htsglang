# #349 boot matrix — sweep 1 findings (2026-08-01)

First real run of the 17-arm cross-feature bug net. Vehicle Qwen3.6-27B-FP8,
TP=3 uneven (5090 + 2x 3080), verdicts produced by
`sglang.srt.boot_matrix.check.check_arm` from the collected artifacts — the
same core the tenant uses, so the sweep and the tenant cannot disagree.

Artifacts and the per-arm table:
`/spinning/gpu-battery-results/2026-08-01_349_boot_matrix/` (`RESULTS.txt`,
plus `arm.json` / `server.log` / `probes.json` per arm).

**Tally: 5 PASS, 12 FAIL.** Wall time about 50 minutes for all 17 arms, not
the estimated 69 — because ten of them died at argument resolution in seconds
instead of booting.

## The headline

Almost every FAIL is a defect in the MATRIX, not in the product. That is a
real result for a first run, and it is the cheap half of the value: a bug net
whose arms cannot boot reports nothing forever, and nothing is exactly what it
would have reported.

The product guards, by contrast, behaved correctly everywhere they fired:
loud, named, at argument resolution, before weight load.

## The gate is inert — the most important finding

**The A-vs-A band is never measured.** The module docstring (`sweep.py:22-27`)
promises that `A_default` is booted first and probed twice, its first-run text
becoming the byte reference and the minimum of the two runs the graded floor.
No such code exists: `main()` calls `run_arm()` at `sweep.py:325` without
`reference_probes`, nothing runs an arm twice, and `reference_probes` is only
ever consumed as `reference_probes or {}` (`:173`).

Consequence: every arm's `probes.json` carries `ref_text: ""` and
`min_score: 0`, including `A_default`'s own. The byte tier has nothing to
compare against and the graded tier's floor is zero, so **the coherence half
of the gate cannot fail**. Every "coherence within the A-vs-A band" in this
sweep is vacuous, and no coherence signal was produced at all.

The probe texts themselves are coherent and identical across `A_default`,
`E_barlink` and `K_bar1_graphs` (byte `4..13`; alphabet `w x y z`; squares
`12 144 .. 20 400`) — reassuring about the product, and exactly the
observation the band was supposed to make rigorous instead of anecdotal.

## `K_bar1_graphs` is a false FAIL, and the crossing is sound

K booted to ready in **131 s against its 1200 s ceiling**, `ACHIEVED=bar1` on
all three barlink groups on all three ranks, both probes coherent. It scores
FAIL only because `check_arm`'s fatal detector matches
`Traceback (most recent call last):` at `server.log:81` — the torchcodec
optional-dependency traceback that sglang itself logs under *"Ignore import
error when loading ...mimo_audio"*. The Docker image ships torchcodec without
a matching ffmpeg, so that benign block appears in **every** containerised
boot; the CT999 arms never see it.

The detector must ignore tracebacks the log itself frames as ignored, or the
whole Docker route reports FAIL forever.

Substantively: bar1 x CUDA graphs works and is fast — 131 s warm against
>18 min cold in #369, which confirms the #370 cold-cache finding from the
other side.

## Arm-definition defects (8 arms)

### The `KVSO_ALLOW_SPEC` gate — B, D, G, I, J

`--enable-kv-session-offload` and `--kv-session-offload-spec-in-tick` are
gated behind `KVSO_ALLOW_SPEC=1` while spill+MTP is in bring-up
(`server_args.py:6165`, `:6179`). Five arms cross offload with speculative
decoding — which is precisely the gated combination — and none of them sets
the env. They cannot boot as written.

Fix: add `KVSO_ALLOW_SPEC=1` to those arms' `env`. The gate is deliberate and
should not be removed; the arms just have to opt in the way an operator would.

### `H_ps2_prefill_spill` — `--speculative-algorithm none` is not a thing

The arm declares "no spec" by passing `--speculative-algorithm none`, and then
dies on the KVSO spec gate with a self-contradicting message:

    --enable-kv-session-offload does not yet support speculative decoding
    (--speculative-algorithm=none). Set KVSO_ALLOW_SPEC=1 ...

`speculative_algorithm` is a free-form `Optional[str]` defaulting to `None`
(`server_args.py:2920`) with no choices list and no normalisation of the
string `"none"`. So `--speculative-algorithm none` does not disable
speculation — it names an algorithm called "none", which is truthy at every
`if self.speculative_algorithm:` site (e.g. `:11409`).

Two separate things to fix, and they are not the same size:

- **The arm** (small): to run without speculation, omit the flag. This is the
  actual reason the arm failed.
- **The product** (worth a ticket): an unregistered algorithm name is accepted
  silently at parse time and first surfaces as an unrelated guard printing a
  contradiction. A name that is not a builtin and not registered should be
  refused by name where it is parsed. Not fixed here — it is not this
  ticket's mandate and it deserves its own test.

### `C_crossalgo` — a required companion flag is missing

`--speculative-cross-algorithm` needs `--speculative-cross-algorithm-force`;
the flag's own help text says "(required)" (`server_args.py:3088`), the
default is `None`, and `parse_cross_force` refuses `None` by name. The arm
passes the feature flag and the lazy-capture flag but not `--force`.

Fix: give the arm a `--force` value. `nextn` is the least exotic.

### `L_video_cotenancy` — same shape

`--dual-group-lane requires --dual-group-lane-budget-mib (there is no fallback
to --mem-fraction-static)`. The arm sets the lane and not the budget.

## Matrix defects in the reject arms

**The answer to "did all 6 reject arms reject cleanly" is no — only 2 did.**

| arm | verdict | what actually happened |
|---|---|---|
| `reject_dcp_topk` | PASS | refused by the real DCP guard. Genuine. |
| `reject_dcp_multilayer` | PASS | refused by the real DCP guard. Genuine. |
| `reject_dcp_crossalgo` | PASS | **false pass** — refused because `--force` was missing, not because of DCP x cross-algo. Marker `("cross-algorithm",)` matched the wrong sentence. |
| `reject_dcp_offload` | PASS | **false pass** — refused by the `KVSO_ALLOW_SPEC` bring-up gate, not by DCP x offload. Marker `("--enable-kv-session-offload",)` matched the wrong sentence. |
| `reject_dcp_offlane` | FAIL | refused, but with an error other than the expected guard. |
| `reject_dcp_draftextend` | FAIL | **booted to ready.** |

The two false passes are the more dangerous finding. `reject_markers` is a
substring test against the whole refusal text, so an arm can report PASS while
never exercising its crossing — a reject arm that dies earlier, for an
unrelated reason, looks identical to one whose guard fired. A reject arm
should assert the guard it names, not any sentence containing its flag.

### `reject_dcp_draftextend` is stale, and the product is right

The arm expects `--draft-kv-layout dcp` alone to be refused with a
"draft-EXTEND" marker. It boots. That refusal was **deliberately removed by
#108 slice 2** — `server_args.py:6978` records that the draft-EXTEND uneven-DCP
metadata split now exists (`flashinfer_backend.call_begin_forward`,
`EAGLE_DRAFT_EXTEND` branch), so the blanket "not usable yet" refusal from
slice 1 is gone and the layout is admitted on the weighted lane with a
one-layer chain draft.

So the configuration is legitimately bootable now and the arm was never
updated. Delete the arm or re-point it at a combination that is still refused.

## `E_barlink` — an effective-config reader mismatch

    resolved a configuration it did not declare -- barlink: declared 'device',
    resolved 'up'

The arm booted and served; what failed is the comparison. `effective.py` reads
the barlink axis out of the log and gets the string `up` (the communicator's
state word) where the arm declares the transport name `device`. Either the
reader should extract the transport name, or the arm should declare the state.

This one matters more than it looks: reading the effective config FROM THE LOG
is the whole #340 lesson this matrix is built on. A reader that returns the
wrong field for an axis will mis-grade every arm that crosses that axis — and
it is grading against declarations nobody re-checked after the barlink rename.

## What actually passed

`A_default` — booted, resolved as declared, coherence inside the measured
A-vs-A band. The baseline is sound, the band mechanism works, and the byte +
graded gate does what it says.

## Recommended order of work

1. The A-vs-A band must actually be measured (`sweep.py:325` never passes
   `reference_probes`); until then the coherence gate cannot fail.
2. `reject_markers` must assert the named guard, not a substring of the whole
   refusal (2 false passes hide behind this today).
3. Fix the seven arm definitions (env opt-in x5, `--force`, lane budget).
4. Fix or retire `reject_dcp_draftextend` (#108 slice 2 made it valid).
5. Fix the `E_barlink` axis reader in `effective.py`, and make the fatal
   detector ignore tracebacks the log itself frames as ignored (today a false
   FAIL on every Docker boot).
6. Ticket the `--speculative-algorithm <unregistered name>` validation gap.

Only after 1-5 does a green matrix mean anything, and only then is a re-run
worth the card time.
