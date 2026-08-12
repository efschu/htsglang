# The #485 confound, settled: two failures, two different causes

Successor 49, 2026-08-12. Prediction recorded before the boot in
`PREDICTION_confound.md`; **it was wrong on the failure it was aimed at**, and
the way it was wrong is the finding.

## The experiment

N48 set `SGLANG_UNEVEN_TOKEN_VECTOR` per arm to that arm's attention split, so
both of arm C's failures were reachable from either the CUT or the TOKEN
VECTOR. One boot separates them: arm C's cut with the SHIP vector.

`boot_c_shipvec_340000.log` is byte-identical to N48's wedging `boot_c.log`
except for one environment variable.

| | N48 `boot_c.log` | this boot |
|---|---|---|
| `--pp-stage-ratio` | 42,11,11 | 42,11,11 |
| `--pp-attn-stage-ratio` | 10,3,3 | 10,3,3 |
| `--max-total-tokens` | 340000 | 340000 |
| `--rank-gpu-memory-mib` | 31400,19300,19300 | 31400,19300,19300 |
| flip | on | on |
| **`SGLANG_UNEVEN_TOKEN_VECTOR`** | **10,3,3** | **7,5,4 (ship)** |

## Result 1 — the seam wedge follows the TOKEN VECTOR, not the cut

| | N48, vector 10,3,3 | this boot, vector 7,5,4 |
|---|---:|---:|
| flips completed | 3 | **6** |
| flips abandoned | 185 (555 log lines) | **0** |
| reached `/health` 200 | never | **yes** |

The wedge is **gone** at the same cut, same pool, same budgets. The planner
cut is NOT locked out of the flip. C34's headline — "every cut that moves the
attention split off `[7,5,4]` starves the seam staging" — is **too strong**:
what starves the seam staging is moving the KV ARENA off the ship vector, and
the arm set moved both at once.

**My recorded prediction said the opposite.** It derived the seam demand from
`_staging_bytes` as `incoming + max(outgoing, local)` over the attention split
`a_i`, got 10 units for arm C's rank0 against 8 for arm A, and called the cut.
The derivation correctly predicts WHICH RANK refuses in both of N48's wedges
(rank0 for arm C, rank2 for arm D) and still gets the CAUSE wrong, because the
row counts in that formula are set by the arena — the token vector — and the
per-arm vector moved with the cut in every arm N48 ran. A model that predicts
the right rank for the right reason can still be attributing the wrong
variable when the variables were never separated.

## Result 2 — the pool-accounting crash follows the CUT, and is a real bug

The same boot then died on the first deep prefill, with the ship vector:

```
ValueError: pool memory leak detected! [full] total=340000,
  available=258360, evictable=0, protected=0, session_held=0,
  uncached=0, withheld=163280
```

Put beside N48's:

| boot | pool | vector | total-available | withheld | ratio |
|---|---:|---|---:|---:|---:|
| N48 arm C | 280000 | 10,3,3 | 12783 | 25566 | **2.000** |
| this boot | 340000 | 7,5,4 | 81640 | 163280 | **2.000** |

`withheld == 2 x (total - available)` **exactly**, across two pools and two
token vectors. An exact factor of two that survives both axes is a duplicate
booking, not a share or rounding artifact — and that arithmetic is what
identified the defect.

**Root cause, reproduced against the real class:** `KvRowCap._apply`
accumulates `_withheld`, and it was wired as the allocator's on-CLEAR hook as
well as its on-free hook (`managers/kv_backing_relief.py`). `clear()` rebuilds
`free_pages = arange(1, size+1)`, so the ids above the cap are taken a SECOND
time and concatenated onto the ids the cap already believes it holds. The free
list stays correct — which is why `available` is right — and only the
published counter doubles. The idle invariant then reports a leak on an
intact pool.

The worse half: `release()` cats that doubled tensor straight back into
`free_pages`, so the duplicated ids are handed out twice — two requests
writing the same KV row, silently. The crash was the lucky outcome.

**Why the ship cut never sees it.** `KvRowCap` only exists once the KV relief
rung engages, which needs a corridor deficit. Attention `10,3,3` puts 10 of 16
attention layers on rank0, ~1.43x the ship cut's 7, so the same pool costs
~43% more VRAM there, breaches the corridor, engages the cap — and only then
is doubling visible. Doubling zero is invisible. Measured on this boot: rank0
(nvidia-smi idx 1) fell to **608 MiB free, 135 samples under the 1024 law**,
while N48's control arm held **7212 MiB** on the same card and budget. The
corridor breach and the cap engagement are the same event.

## What this changes

* **C34 is reframed, not resolved.** The cut is not locked out by the seam.
  Two separable defects were being read as one wall.
* **Law 23 stands and is strengthened.** The residency gate still had no
  transient term — but the thing that actually stopped the cut booting was
  neither residency nor staging: it was an accounting bug that only a
  corridor-deficit configuration can reach. "Fits at rest" and "can run"
  differ by more mechanisms than the seam.
* **A method note.** N48 varied cut and vector together because the vector
  "should" follow the attention layers. That is good physics and bad
  experiment design: it made two failures share one explanation for a whole
  shift. When a derived setting rides along with the variable under test,
  the arm set has two variables in it.
