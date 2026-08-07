# DESIGN_631c — dual-group boot: composition verdict, stride falsifier, recipe

Follows `DESIGN_631b_draft_kv_wiring.md` (ec3a44003e). That document specified
the draft-KV wiring and left three items owed at the desk. This one discharges
them. Everything below was EXECUTED on a cardless box
(`CUDA_VISIBLE_DEVICES=99`); nothing here is a reading argument alone.

Probes: `scratch_631/probe_step1_composition.py`,
`scratch_631/probe_step1b_lift.py`, `scratch_631/probe_step3_recipe.py`.
Falsifier: `test/registered/unit/disaggregation/test_pd_draft_kv_stride_631.py`.

Two harness stubs are used throughout and neither is a gate under test:
`is_cuda()`/`get_device()` (this box has no accelerator) and
`_resolve_rank_gpu_cards` (fabricates the rig's three cards, since #392
correctly refuses to guess a card identity it cannot resolve).

## 1. Composition verdict: zero reachable configurations

DESIGN_631b §0b recorded the combination as "unconsidered". That is now out of
date in one direction and understated in another.

**Out of date:** #642 (`66ca5eed12`) landed after it.
`arg_groups/pd_disaggregation_hook.py:96 validate_pd_draft_kv_layout` now
requires `--draft-kv-layout dcp` on a token-sharded PD arm carrying
speculation. The combination is considered.

**Understated:** it is not merely unsupported, it is UNREACHABLE. The gate
order in `ServerArgs.__post_init__` is

| line | gate |
|------|------|
| `server_args.py:5916` | `_handle_pd_disaggregation` → **#631a refusal** |
| `server_args.py:5922` | `_handle_dcp_validation` |
| `server_args.py:5957` | `_handle_uneven_tp` (resolves `dcp_size`) |
| `server_args.py:6067` | `handle_speculative_decoding` (resolves the alias) |
| `server_args.py:6113` | `_reject_unsupported_draft_kv_dcp` (#108) |
| `server_args.py:6114` | `_validate_pd_dcp_token_shard_contract` (#636) |
| `server_args.py:6123` | `_validate_pd_draft_kv_layout` (#642) |

#631a raises at `:5916`, before all three later gates. Measured matrix:

| cell | outcome |
|------|---------|
| monolithic + spec + `replicated` | ACCEPTED |
| monolithic + spec + `dcp` | ACCEPTED |
| PD prefill/decode + spec + either layout | REFUSED by #631a at `:5916` |
| PD decode + spec + `dcp` + `SGLANG_PD_AUTO_DISABLE_SPEC=1` | REFUSED by #108 — "requires ... a speculative algorithm ... Got `speculative_algorithm=None`" |
| PD decode + no spec + `dcp` | REFUSED by #108, same text |
| PD decode + no spec + `replicated` | ACCEPTED |

So `--draft-kv-layout dcp` composes with `disaggregation_mode` in **exactly
zero** reachable configurations, and **#642 is dead code today** — no input
reaches it with `speculative_algorithm is not None`. Its docstring claims this
deliberately; the matrix proves it rather than asserting it.

One misleading refusal falls out: under the `SGLANG_PD_AUTO_DISABLE_SPEC`
escape hatch the operator asks for `dcp` WITH speculation, the hatch removes
the speculation at `:5916`, and the #108 gate at `:6113` then blames the
operator for not supplying one. A correct gate would notice that
`speculative_algorithm` was nulled by the hatch rather than omitted.

## 2. What a correct gate must additionally check

Simulating the #631a lift (`probe_step1b_lift.py`, which hides
`speculative_algorithm` for the duration of the PD hook and nothing else)
re-runs the matrix with the refusal removed:

| cell | outcome with #631a lifted |
|------|---------------------------|
| decode tp=3/dcp=3 + spec + `dcp` | ACCEPTED |
| decode tp=3/dcp=3 + spec + `replicated` | REFUSED by #642 — its can-fail arm firing |
| **prefill pp=3/tp=1 + spec + `replicated`** | **ACCEPTED** |
| **prefill pp=3/tp=1 + spec + `dcp`** | **REFUSED by #108** |
| prefill pp=3/tp=1 + no spec | ACCEPTED |

The middle two rows are the finding. On the #631 topology the two arms are
**forced onto different `--draft-kv-layout` values**:

- the PP prefill group (tp=1, dcp=1) cannot take `dcp` — the #108 gate demands
  a non-uniform `--rank-tp-ratio` and `dcp_size == tp_size > 1`, and a PP group
  has neither;
- the token-sharded decode group cannot take `replicated` — #642 refuses it;
- and #642 is **structurally inert on the prefill arm**, because its
  precondition `dcp_size > 1 and dcp_size == tp_size`
  (`pd_disaggregation_hook.py:142-146`) is false on a tp=1 group.

Therefore DESIGN_631b's **L10 ("both arms run the SAME `--draft-kv-layout`")
is not merely unbuilt — it is unsatisfiable as written on this topology.** A
correct gate cannot be a per-arm parse-time check at all: parse time cannot
see the peer. It has to be a handshake-time comparison, which is why the
`DraftKvCanonicalLayout` version channel (`214433ec2b`) should not be retired.
What it must compare is not the layout NAME but the resulting geometry —
per-layer item length and row count — because on this topology the names
legitimately differ while the geometry may still agree.

## 3. The stride falsifier, and the defect it found

DESIGN_631b §0a settled owed item (1) by construction: under `dcp` the draft
pool carries `get_total_num_kv_heads()` exactly as the target does, so per-rank
item lengths agree. That reasoning is correct about head counts and does not
reach the two places the boundary actually reads.

### 3a. The stride guard is one index wide

- `mooncake/conn.py:2114` — the decode arm advertises **one scalar**,
  `kv_args.kv_item_lens[0]`, for its entire registration. The appended draft
  layers' item lengths are never transmitted. `KVArgsRegisterInfo.dst_kv_item_len`
  (`conn.py:134`) is a single `int`.
- `mooncake/conn.py:1517-1528` — the only stride guard compares that scalar
  against the prefill arm's `kv_item_lens[0]`. Also index 0.

A divergence confined to the appended draft layers passes it. The falsifier
carries both arms: `test_guard_catches_a_target_stride_divergence` (layer-0
skew, guard fires) and `test_guard_misses_a_draft_only_stride_divergence`
(draft-only skew, guard silent).

### 3b. The real defect: PP prefill + a draft pool mispairs buffers silently

`common/conn.py:736 get_mha_kv_ptrs_with_pp` recovers K and V pointers by
splitting the flat registration list in half. The target pool registers
`k_buffer + v_buffer` (`memory_pool.py:2483`), so the halves are exact.
`prefill.py:186-195` / `decode.py:439-448` then APPEND the draft pool to that
same flat list, and the split has no notion that they are there.

When both arms have equal-length lists the mislabel is **symmetric** and the
pairing survives — which is why this is invisible on a non-PP pair, and why
the falsifier records that case as a passing control. Under PP it is not
symmetric: the prefill stage registers `2*stage_layers + 2` entries against the
decode arm's `2*total_layers + 2`, the `elif` branch at `common/conn.py:748`
computes `multiplier_ratio` from those two, and the destination slices land in
the wrong section of the list.

Measured on a 48-layer model over three PP stages, draft pool present on both
arms:

| PP stage | mispaired buffers | example |
|----------|-------------------|---------|
| start_layer=0 | 18 of 34 | src `V0` → dst `K16` |
| start_layer=16 | 18 of 34 | src `V16` → dst `K32` |
| start_layer=32 | 18 of 34 | src `V32` → dst `V0` |

**Control, identical geometry with the draft pool removed: 0 of 32 mispaired
on every stage.** That control is what makes the result evidence rather than a
harness artefact. No exception is raised on the mispaired path — it returns
normally. This is #345's class: right token, wrong slot, no crash.

DESIGN_631b's W1 REQUIRES the combination that triggers it ("the prefill arm
must LOAD the draft layer's weights"), and `prefill.py:164-165` appends the
draft pool on every stage whenever layer-sharding is off, which PP does not
turn on. So the #631 topology walks straight into it. **This is the ticket**,
and it is upstream of L1-L9: no amount of layout negotiation fixes a pointer
partition that is wrong before the layout is consulted.

Arm 1 is unaffected — with speculation refused there is no draft pool, the
list lengths are `2*stage_layers` against `2*total_layers`, the `else` branch
runs, and the control arm shows it pairing exactly.

## 4. Boot recipe

### 4.0 Regime, first — co-residency is not the default shape

Operator ruling (2026-08-07): **co-residency of two weight-holding groups is
only viable while little KV is in demand. Under real KV pressure the correct
move is a full reshard with maximum possible reuse, parking the remainder in
system RAM.** So a static two-group `--rank-gpu-memory-mib` partition is the
wrong shape for anything but a bring-up. Arm 1 below is explicitly a low-KV
bring-up (16k context, 8 concurrent requests) and must not be read as the
serving topology. The steady-state design belongs to the reshard/park
machinery (#286/#330 asset register, #407 tier registry), not to a fixed
split.

`--disaggregation-topology colocated-congruent`
(`disaggregation/topology.py:28-36`) is worth naming here: a card keeps its
decode rank AND serves the prefill lane FROM THE SAME PROCESS on the decode
rank's resident shard — "co-location costs activations and scratch, not a
second model copy". It sidesteps the second weight copy entirely. It does not
deliver #631's premise, because the prefill lane is then congruent with the
decode TP layout rather than a PP group, and PP-beats-TP-on-prefill is the
whole measurement that motivated #631. Recorded as the cheap fallback if the
PP group proves unaffordable.

### 4.1 Named blocker for the literal ask

A PP prefill group (pp=3) and a TP decode group (tp=3) both resident on the
same three cards is `--disaggregation-topology colocated-process`. This rig
refuses it on two independent counts, both verified:

- **No MPS daemon.** `/tmp/nvidia-mps` does not exist.
- **NCCL below 2.30.** The venv ships `nvidia-nccl-cu13 2.28.9` and
  `nvidia-nccl-cu12 2.29.7`.

`topology.py:684-707` gates `colocated-process` on exactly those two probes.
There are only three cards, so `disjoint` cannot give both groups ≥2 cards
either. The unblock is the Docker image (`docker/htsglang.Dockerfile` pins
NCCL 2.30.7) plus starting an MPS daemon — an operator decision, not mine.

**Warning:** ARM 2 below parses fine WITHOUT `--disaggregation-topology`.
#107's per-card VRAM feasibility check only runs when that flag is passed, so
omitting it means two servers each claim all three cards with no cross-process
accounting, and the failure is a runtime OOM rather than a boot refusal.
Always pass the topology flag for a co-resident pair.

### 4.2 ARM 1 — runnable today, proves the main handover healthy

Disjoint: no shared card, so neither the NCCL threshold nor MPS applies.
Prefill is a genuine PP group (pp=2) on the two 3080s; decode is tp=1 on the
5090. `Qwen3.5-2B` is the same `Qwen3_5ForConditionalGeneration` hybrid
architecture as the 27B (linear_attention + full_attention every 4, head_dim
256), so it exercises the same GDN-state + KV transfer paths at 3.5 GB instead
of 25 GB. Speculation is absent on both arms by construction (#631a), which is
the "two unknowns do not get mixed" rule.

All four command lines below were parsed literally by
`scratch_631/probe_step3_recipe.py` and accepted.

```bash
# GPU arbitration first -- take the holder and start the heartbeat before any
# card is touched. Stop the heartbeat BEFORE releasing.
#   see /spinning/gpu-arb/

export PYTHONPATH=/spinning/wt-631-dualgroup/python
export LD_LIBRARY_PATH="/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
PY=/spinning/htsglang-gpu/.venv/bin/python
MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-2B

# --- PREFILL arm: PP group (pp=2) on the two 3080s (CUDA ordinals 1,2) -----
setsid $PY -m sglang.launch_server \
  --model-path $MODEL \
  --served-model-name pd631-arm1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --pp-size 2 \
  --tp-size 1 \
  --base-gpu-id 1 \
  --page-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 16384 \
  --max-running-requests 8 \
  --mem-fraction-static 0.60 \
  --trust-remote-code \
  --enable-metrics \
  --host 127.0.0.1 --port 31241 \
  > /spinning/wt-631-dualgroup/scratch_631/arm1_prefill.log 2>&1 &

# --- DECODE arm: tp=1 on the 5090 (CUDA ordinal 0) -------------------------
setsid $PY -m sglang.launch_server \
  --model-path $MODEL \
  --served-model-name pd631-arm1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --tp-size 1 \
  --base-gpu-id 0 \
  --page-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 16384 \
  --max-running-requests 8 \
  --mem-fraction-static 0.60 \
  --trust-remote-code \
  --enable-metrics \
  --host 127.0.0.1 --port 31243 \
  > /spinning/wt-631-dualgroup/scratch_631/arm1_decode.log 2>&1 &

# --- gate, then drive ------------------------------------------------------
$PY /spinning/wt-631-dualgroup/scripts/satellite/prefill_offload.py preflight \
  --prefill http://127.0.0.1:31241 --decode http://127.0.0.1:31243

$PY /spinning/wt-631-dualgroup/scripts/satellite/prefill_offload.py measure \
  --prefill http://127.0.0.1:31241 --decode http://127.0.0.1:31243 \
  --bootstrap-host 127.0.0.1 --bootstrap-port 8998 \
  --load 3 --prompt-tokens 8192 --seed 1631
```

Notes bound to this recipe:

- `--page-size 1` and a matching `--kv-cache-dtype` on both arms are the #636
  contract, not style. The PD handshake compares only those two, which is why
  the weights check has to come from elsewhere — see the next note.
- **The #212 model-mismatch preflight is now covered in-tree.** Guard 1
  (`57e13eb9d3`) is genuinely wired, not merely declared:
  `disaggregation/common/conn.py:482-491` computes the local
  `compute_model_identity_hash`, compares it to the peer's and refuses on
  mismatch. Verified at the caller, per the R6 ledger rule. Run the external
  `preflight` anyway — it is cheap and catches port/URL errors the hash cannot.
- **`--disagg-decode-enable-radix-cache` is deliberately absent, but the
  briefing's stated reason does not hold at this commit.** There is no
  Mamba/SSM `ValueError` for it: `pd_disaggregation_hook.py:231-247` refuses it
  only for `--enable-hisparse`, the `fake` backend, and speculation. The real
  hazard is the one the runbook records — `MambaRadixCache._match_post_processor`
  truncates a prefix match to the deepest node owning a mamba checkpoint, so a
  KV-only match returns zero tokens and it silently does nothing. Omit the flag;
  do not expect a refusal to protect you.
- Disk HiCache is omitted on purpose. It is mandatory for serving boots; these
  are bring-up arms, and PD × HiCache is a separate axis that would confound
  the handover reading.
- `--rank-tp-ratio auto-performance` is REFUSED under `--pp-size > 1`
  (`server_args.py:9850`) — use plain `auto` or an explicit vector on any PP
  group.
- Never pass `--dcp-size` explicitly alongside `--speculative-algorithm NEXTN`:
  `_handle_dcp_validation` runs at `:5922`, before the alias resolves at
  `:6067`, and dies with "Unknown speculative algorithm name: NEXTN". Let
  `SGLANG_UNEVEN_DCP=1` auto-set it in `_handle_uneven_tp`.

### 4.3 ARM 2 — the literal ask, blocked on §4.1

Arguments verified well-formed; the environment is what refuses. Add
`--disaggregation-topology colocated-process` (and its
`--disaggregation-prefill-layer-split`) once MPS and NCCL ≥ 2.30 are present.

```bash
export SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1

# PREFILL: PP group (pp=3) across all three cards
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --served-model-name pd631-arm2 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --pp-size 3 --tp-size 1 \
  --rank-gpu-id 0,1,2 --rank-gpu-memory-mib 13000,8000,8000 \
  --page-size 1 --kv-cache-dtype fp8_e4m3 --context-length 16384 \
  --max-running-requests 8 --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port 31251

# DECODE: TP group (tp=3, uneven weighted DCP) across all three cards
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8 \
  --served-model-name pd631-arm2 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tp-size 3 --rank-tp-ratio 3,2,2 --rank-kv-ratio 3,2,2 \
  --rank-gpu-id 0,1,2 --rank-gpu-memory-mib 17000,10000,10000 \
  --page-size 1 --kv-cache-dtype fp8_e4m3 --context-length 16384 \
  --max-running-requests 8 --trust-remote-code --enable-metrics \
  --host 127.0.0.1 --port 31253
```

The MiB vectors above are a starting partition, not a measured fit: per card
they sum to 30000 / 18000 / 18000 against NVML totals of 32768 / 20480 / 20480,
leaving 2768 / 2480 / 2480 MiB for two CUDA contexts plus the 1024 MiB user
corridor per card. That corridor is measured as the **NVML FREE column, never
total minus used** (the carve-out of ~424-518 MiB is invisible to
total−used). Whether this actually fits is a window question, and per §4.0 it
is the wrong shape for anything beyond bring-up anyway.

## 5. Is L1-L9 reachable in one window?

No. Two windows minimum, and the second is not schedulable yet.

Window 1 can do arm 1 end to end: boot the disjoint pair, prove the main
handover healthy on the hybrid GDN model, and establish the A-vs-A noise floor
that L6 will later be measured against. That is a full window's work and it is
worth having on its own.

L1-L9 cannot follow in the same window for three independent reasons, any one
of which is sufficient:

1. **§3b is a defect, not a wiring step.** The PP-plus-draft pointer
   mispairing has to be fixed and re-falsified before any draft byte crosses
   the boundary. That is a code change with its own review.
2. **L10 is unsatisfiable as specified (§2).** The requirement has to be
   redesigned into a geometry comparison at the handshake before it can be
   demonstrated.
3. **The topology L1-L9 needs does not boot on this rig (§4.1).** MPS plus a
   NCCL 2.30 runtime is an environment change, and arm 1 deliberately avoids
   needing it.

L6 in particular wants a same-boot A-vs-A floor plus a monolithic comparison
arm, which is most of a window by itself. Planning L1-L9 as a tail on window 1
would produce exactly the skipped-L6 outcome DESIGN_631b warns about.
