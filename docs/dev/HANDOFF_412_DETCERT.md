# HANDOFF #412 — determinism certificate mode

Branch `feat/deterministic-hetero-412`, worktree `/spinning/wt-412-detcert`,
based on `origin/feat/route-a-631` @ `3be93fa943`. One commit: `c58bbaf74f`.
**No merges performed** — operator sequences.

Run tests with `PYTHONPATH=/spinning/wt-412-detcert/python` and
`/spinning/htsglang-gpu/.venv/bin/pytest`. Without that PYTHONPATH the suite
tests `/spinning/htsglang-gpu/python` instead and the result is meaningless.

---

## ERRORS FIRST

### 1. The GPU byte-gate was NOT run. This is the open acceptance step.

Arbitration was held continuously by the parallel #656 strand for the whole
shift (`/spinning/gpu-arb/holder` = `656-successor50`, ACTIVE since
2026-08-12T10:42Z, serving live on 30030 from `/spinning/wt-631-routea`;
heartbeat fresh within seconds on every check). No cards were taken and nothing
of theirs was touched. The desk deliverable stands alone; the executable recipe
is `docs/dev/DESIGN_412_determinism_certificate.md` §6 — arms, seed discipline,
what gates (A1/A2/B/C) and what is recorded only (D/F, and **E cross-boot is
never a gate**).

### 2. Five `tests/moe_offload` tests fail — environmental, not this change.

3× `torch.OutOfMemoryError`, 1× `CUDA error: operation not supported`, 1×
`AttributeError: 'object' object has no attribute 'record_event'` raised inside
`torch/cuda/streams.py:77` against a test mock. The 5090 had ~19 GB in use by
the other strand's serving. This branch's only `expert_offload.py` change is a
docstring. Re-check them in a quiet window before attributing anything to #412.

### 3. Two register claims are WRONG and the code now contradicts them in source.

Both were repeated across handoffs and are load-bearing, so they will bite
again if not corrected at the source of the prose:

- **"fa3 is Hopper-only and does not boot on these SM86 cards"** — N44
  (`HANDOFF_688.md:117-120`) and the N45 block
  (`CONTRADICTIONS_REGISTER.md:259-267`). False.
  `sgl-kernel/python/sgl_kernel/flash_attn.py:15-28` accepts compute-capability
  major **8 or 9** with CUDA ≥ 12.3, comment naming sm80/sm86/sm89/sm90a.
  fa3 runs on a 3080; it is **sm120** that it refuses, and the raise there reads
  "only supported on sm90 and above" (`flash_attn.py:306-309`), which is
  presumably how the inversion survived. Pinned by
  `test_fa3_window_excludes_sm120_and_includes_sm86`.
- **"~109 tokens is GDN long-prefill nondeterminism"** — the brief for this task
  said so; the memory register has already self-corrected
  (`gdn-prefill-nichtdeterminismus.md:3`: "KORRIGIERT #190 … NICHT GDN — GDN-Lane
  bitweise clean"). Root cause is `gptq_marlin_gemm` on sm80..88, scope is every
  fp8 checkpoint routing through Marlin there, model-independent. Measured table:
  `INTEGRATION_R3_VALIDATION.md:4218-4224` (0/1200 through M=109, first at M=128,
  **non-monotonic above** — M=129/160/205 are clean again).

### 4. `"#225"` does not exist anywhere in this tree.

The brief cites it for the spec/temp-0 finding. Grep returns nothing in `docs/`
or `python/`. The substantive measurement is
`INTEGRATION_R3_VALIDATION.md:3476-3506` (#143 Window 5) plus
`FEATURES_VS_UPSTREAM.md:27`. The exclusion cites those, not the ticket number,
so a reader chasing it lands somewhere real. **Still open; no fix, no gate.**

### 5. A false claim was shipped in source and is now corrected — check for others.

`layers/moe/expert_offload.py`'s module docstring claimed the offloaded path is
"bit-identical to the no-offload (fraction == 1.0) path", unqualified. The
module's own marlin apply (`:2898-2902`) sets `layer.num_local_experts =
buf_slots`, changing `moe_align_block_size`'s `global_num_experts` → GEMM tiling
→ reassociation (~1e-2 on marlin int4; sub-ULP but non-zero on fp8, whose
"byte-identical" claim the author retracted after a 256-token re-run agreed
118/256). Now scoped to the wave mechanism, pointing at the module's two
correctly scoped claims (`:437-442`, `:3615-3622`). **A certificate citing the
old wording would have shipped a false claim** — that is exactly the failure
mode #412 exists to prevent, found inside our own tree.

### 6. Known gap the mode names but cannot close: spilled sessions are invisible.

`req.kv_spill_state` exists server-side (`schedule_batch.py:759`) but never
reaches `meta_info`, so a client cannot distinguish a spilled (uncertified)
answer from a resident (certified) one. A caller can pre-empt it with
`spill_class="never"` (`entrypoints/openai/protocol.py:471-476`), which the
certificate surfaces as a note. **Surfacing `kv_spill_state` in `meta_info` is
the natural follow-up** and would make the per-request exclusion checkable by
the client rather than merely documented.

---

## What landed

| file | what |
|---|---|
| `python/sglang/srt/determinism_certificate.py` | new. Pure resolver (`resolve_certificate`), exclusion library with cites, `GuaranteeClass`, group backend selector, NVML group probe |
| `python/sglang/srt/server_args.py` | `--deterministic-hetero` flag + `_handle_deterministic_hetero`, called immediately **before** `_handle_deterministic_inference` so the base mode validates a pinned group-valid backend instead of guessing from one device |
| `tests/determinism/test_certificate.py` | 38 hermetic CPU tests |
| `docs/dev/DESIGN_412_determinism_certificate.md` | guarantee envelope, exclusions with evidence, CI byte-gate recipe |
| `python/sglang/srt/layers/moe/expert_offload.py` | docstring correction (item 5) |

**Design contract:** REFUSE only impossibilities; narrow-and-name everything
else. `resolve_certificate` is pure — no torch, no NVML, no env — so the whole
decision surface is testable on CPU; the probe is a thin adapter.

## Test status

- `tests/determinism`: **102 passed** (64 pre-existing + 38 new), ~7 s.
- ruff: clean on every touched file (`server_args.py`'s 714 findings are
  pre-existing repo-wide and untouched).
- **Falsification sweep run** — six mutations, each turning the intended test red
  and only that test, baseline restored green after each: fa3 window narrowed to
  Hopper-only (4 red), selector pinned to flashinfer (3 red), broadcast refusal
  removed (1 red), spill exclusion widened to boot scope (1 red), fp8 env armed
  unconditionally (1 red), handler ignoring its own flag (1 red). One earlier
  mutation attempt silently failed to apply (a comment between the `if` and its
  body) and was redone — worth remembering that a green sweep step can mean the
  mutation missed, not that the guard is weak.

## Next

1. Take the byte-gate at the next free window (§6 of the design doc). Diff the
   block a live server prints against
   `test_guarantee_statement_pins_the_ship_envelope`; if a real boot's envelope
   differs from the resolver's, one of them is lying.
2. Fill `matrix.py` rows with measured bands (`pending_calibration=True` until
   then — a guessed band can be two orders wrong; **tighten, never loosen**).
3. Surface `kv_spill_state` in `meta_info` (item 6).
4. Correct the N44/N45 prose in the register itself (item 3) — the code now
   disagrees with it in two places, but the prose is what the next shift reads.
