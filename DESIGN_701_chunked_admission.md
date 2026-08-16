# DESIGN 701 — the chunked-prefill self-deadlock

PRIO. Deeper root beneath the #698 wedge. Tree `/spinning/wt-602-slot2`.
Desk design; no GPU touched.

---

## 1 — The specimen, and what it rules out

ONE chunked request (new-seq 1, new-token 512, cached 0) drove pool usage
0.95 -> 1.00 with **no retract, no abort, no finish**. So this is not
contention between requests and not a leak: a single request walked the pool to
full on its own. Any explanation that needs a second actor is excluded by the
specimen.

Two properties make it terminal rather than merely tight:

* the request is **resident-but-batchless** (#631 defect O), so `running_bs`
  reads 0 while its locked prefix holds the pool. Every consumer of `running_bs`
  concludes the instance is idle;
* eviction **cannot** free the chain, because the chain is locked by the very
  request that is trying to grow it. The #698 relief now correctly REPORTS
  `freed 0` — it made the failure legible, and could not fix it.

## 2 — Root cause, in one sentence

**Chunked prefill bounds the COMPUTE per step, not the KV COMMITMENT — and
admission was reading the compute bound as if it were a memory decision.**

`schedule_policy.py:1389-1407`, the chunked branch:

```python
if self.rem_chunk_tokens <= 0:
    return AddReqResult.OTHER
trunc_len = self.rem_chunk_tokens
...
self._update_prefill_budget(0, trunc_len, 0, ...)
```

The budget is charged `trunc_len` — this chunk, 512 tokens. But admitting the
request commits the pool to its **entire remaining length**, because each
subsequent chunk locks more prefix and the locked prefix is never releasable
while the request lives. A 327,680-token request is therefore admitted on the
strength of a 512-token affordability check: a 640x under-charge.

The non-chunked branch above it charges `req.extend_range.length`, the real
figure. Only the chunked path substitutes the chunk for the commitment.

## 3 — Design question (a): fund up front, or spill the own prefix

**Decision: fund the full remaining length at admission. Spill comes later, and
is not a prerequisite.**

The two candidates:

| option | correctness | throughput | cost |
|---|---|---|---|
| **fund full remaining length** | deadlock-free by construction | poor for near-capacity requests: at ctx 327,680 against a ~437k pool only ONE can be in flight | small, local to admission |
| spill/retract the request's OWN prefix (kvso spill, or retract preserving prefix on host) | also deadlock-free | much better concurrency | large: touches the resume/identity path, whose flag is DEFAULT OFF pending determinism work |

Funding first is the right slice order: it is a **correctness** fix that cannot
regress into a wrong answer, and it does not foreclose the spill path — a
future spill capability simply raises `evictable_unlocked` and the SAME
admission rule then admits more. The rule is written against pool arithmetic
precisely so that a spill capability plugs in without re-deriving it.

Stated honestly: this fix makes the head-of-line case SLOWER (some requests get
deferred that today get admitted and then wedge the instance). Trading
throughput for not-deadlocking is the correct trade, and #698's `freed 0` is
the evidence that the current behaviour is not a throughput win but a stall.

## 4 — Design question (b): head-of-line near pool capacity

`context_len 327,680` against a ~437k pool means **one maximum-length request is
~75 % of the pool**. Admission needs a decision for this, and per the binding
generality clause it must come from measured pool arithmetic, not a rig
threshold.

Three verdicts, all derived:

* `required > total_capacity` -> **REFUSE, loudly.** The request can never fit
  at any future time; admitting it can only deadlock. This is the only case that
  is a hard error, and it is derived (capacity), not a tuned fraction.
* `required > free + evictable_unlocked` -> **DEFER.** It does not fit *now* but
  may later, when other requests finish and unlock their chains.
* otherwise -> **ADMIT.**

Note what is deliberately absent: any "90 % of pool" style constant. A request
at 99 % of a large pool is admissible; a request at 101 % of a small one is
refused. Same rule, any hardware.

## 5 — Design question (c): #631 defect O as a counting truth

`running_bs` reading 0 while a chunked request is resident is not a display bug;
it is a false premise handed to every consumer, including the KV-pressure
ladder, the min-free-slots delayer, and the idle/flip detectors (which is how a
genuinely-busy instance armed a flip in the #631 note).

The rule: **a resident-but-batchless chunked request counts as running.** This
design exposes `effective_running_bs(running_bs, resident_chunked)` as the one
place that truth is expressed, so consumers converge on it rather than each
re-deriving it. Wiring every consumer is a follow-up slice; the counting truth
and its test land here so the follow-up has something to converge on.

## 6 — The falsifier

Hermetic, and it must fail against today's arithmetic:

> Admission of a request whose remaining length exceeds
> `free + evictable_unlocked` must be REFUSED or DEFERRED, never admitted.

Plus the specimen itself as a regression: a single 327,680-token request against
a pool with 437k capacity but only ~20k free must not be admitted, and the
decision must not depend on the chunk size — passing a different `chunk_tokens`
must not change the verdict, which is the precise substitution the bug makes.

## 7 — Slice scope

1. `planner/chunked_admission.py` — the rule, as pure pool arithmetic, with the
   three verdicts and a printable reason. No scheduler imports, cherry-pickable.
2. Falsifier tests, red-first.
3. Wiring into `schedule_policy.py`'s chunked branch — **F4-r4 coordination
   required before any deploy**, because it changes admission behaviour on the
   serving line and will defer requests that are admitted today.

Slice 3 is deliberately separated: slices 1 and 2 are inert desk artifacts and
land now; the behaviour change lands with a window.
