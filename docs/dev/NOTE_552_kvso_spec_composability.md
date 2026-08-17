# #552 — kvso × MTP/Spec composability

Desk only, 2026-08-17. No boot, no GPU, no model load. **Nothing built**, and
the reason is the same one that has come up repeatedly on this strand: it is
already built. What is missing is one hardware observation, and the flag help
says so in as many words.

## 0 — The premise: it is not refused, it is opt-in pending a boot

The brief asked why kvso × spec is *refused*. It is not. There are two flags,
both default-OFF, both named decisions:

**`--kv-session-offload-spec-in-tick`** (`server_args.py:2001`) — run the
configured NEXTN/EAGLE drafter *during* the spill tick so a host-resident
session keeps MTP speed instead of a plain bs=1 host decode. Draft KV is kept
device-**resident**; verify is a `num_draft+1`-row target forward that
host-streams the spilled prefix. "The win is FEWER host-KV stream passes per
accepted token (~accept_len factor)."

**`--kv-session-offload-resume-under-spec`** (`server_args.py:2023`) — let a
spilled session wave back and rejoin the LIVE spec decode batch. Its help is
the whole answer to this ticket, verbatim:

> *"Default OFF, and that default is a NAMED decision, not an oversight … The
> resume path is **BUILT** — the draft-KV share spills and restores inside the
> session's residency bundle, the seed is republished into the future map, and
> the host-finish guard lifts at both of its sites — **but it has not been
> observed rejoining a live spec batch on hardware, so it stays opt-in until
> it has.**"*

So the state is: implemented, gated, tested at desk, unobserved on metal.

## 1 — Which of the brief's two cuts the code took: neither, and better

The brief offered (a) discard the draft state and re-enter the spec loop
fresh, or (b) refuse resume-under-spec by name.

The implementation does a third thing: **the draft-KV share spills and
restores WITH the session**, inside the residency bundle
(`kv_session_offload.py:117-122` — the bundle carries `("gdn_state", …)` and
`("draft_kv", …)` entries, and "the draft share spills WITH the" session).

That is stronger than (a). (a) is safe because draft KV is reconstructible;
carrying it is safe *and* avoids paying the reconstruction. There is also a
designed middle path for the tick case — **Option (b')**
(`kv_session_offload.py:157-176`): the tiny 1-layer NEXTN / few-layer EAGLE
draft-KV *tail* stays device-resident while the multi-layer DCP-sharded target
KV spills, so `draft()` runs on device while the session is spilled. Bounded
by `--kv-session-offload-mtp-resident-slices`, explicitly "a QoS knob".

## 2 — The #404 committed-only rule: already the discipline

The brief's constraint — restore from COMMITTED state only, never from
anything a verify touched — is the module's existing discipline, and it was
learned the hard way:

- `kv_committed_len` is bumped **only** in the deferred result processor
  (`kv_committed_len += num_accept_tokens`), and "the result processor remains
  the single writer of `kv_committed_len`" (`kv_session_offload.py:1454-1470`).
- The hazard is recorded with its incident: a stale committed length
  "would span the accepted slots", and `committed=1967 → 1966` once tripped
  the tick-build assert (`:1406`).
- The clock is under the **closed owner register** at
  `test/registered/unit/spec/test_decode_bookkeeping_ownership.py::_OWNER_SITES`
  — an AST scan of the whole `srt/` tree, so any new mutation site turns it
  red, and per `FEATURE_CATALOG.md:2214-2231` "registering to silence the test
  is the one forbidden resolution".

**Consequence for any future work here:** a resume that introduced its own
`kv_committed_len` mutation would have to pull `_OWNER_SITES` along in the same
change, carrying the audit rather than the count. That is the rule that keeps
the #404 lesson enforced rather than remembered.

## 3 — FCFS: the module's stated discipline

`kv_session_offload.py:1` — *"Per-session KV offload to host RAM (S1) — **FCFS
spill / FIFO restore**"*. Victim ordering is `session_priority_key`
(`:976`), whose protection rank is documented as "HIGHER = MORE protected",
with the aggressive mode inert by default (`:939-942`).

So the starvation shape the brief worries about — a younger session's spec
verify overtaking a resumed one — is addressed by FIFO restore plus the
priority key. **NOT ESTABLISHED here:** whether FIFO restore ordering is
preserved once a resumed session re-enters the *live spec* batch, because that
is the path nobody has observed. It is the first thing the window should read.

## 4 — What genuinely stays refused, and why it is by construction

Turning `--kv-session-offload-resume-under-spec` ON **disables** the PS2 deep
prefill spill (`--kv-session-offload-prefill`). Not a policy choice:

> *"a born-spilled prompt never wrote the draft KV that the rejoined session's
> drafter would attend, so the two are mutually exclusive by construction and
> the boot says so."*

That is the #223 decoupling, still holding. It is pinned:
`test/registered/unit/test_kv_session_offload_unit.py:2415`
(`prefill_spill_deep_gate(True, spec_active=True, resume_under_spec=True)`),
alongside the env-twin pins at `:2445` and a dedicated gate suite,
`test/registered/unit/server_args/test_kvso_spec_gate_500.py`.

DFLASH is also excluded for spilled (long-context) sessions — only the one
configured NEXTN/EAGLE drafter is used (`server_args.py:2016-2018`).

## 5 — Why I built nothing

Check-first, after being wrong about exactly this three times this session
(#726, #677, #545): the composability is implemented, the gates are pinned, the
mutual exclusion is pinned, and the committed-only discipline is both
implemented and protected by an AST-scanned owner register. Writing a
"minimal composable cut" would have duplicated shipped machinery, and writing
pins for these properties would have duplicated existing suites — the second
authority problem I refused in #536 and again in #545.

The one thing no amount of desk work can supply is the observation the flag
help names as the reason for the default.

## 6 — Acceptance for the window (filed, not run)

Boot with `--enable-kv-session-offload --speculative-algorithm <NEXTN|EAGLE>
--kv-session-offload-resume-under-spec` (and therefore **without**
`--kv-session-offload-prefill`, which that combination disables by
construction — expect the boot to say so).

1. **A session spills mid-decode under active spec and rejoins the live spec
   batch.** This is the unobserved event; everything else is secondary.
2. **Correctness across the seam:** the resumed session's continuation is
   byte-identical to the same prompt run without spilling, greedy. A resume
   that restored anything a verify touched shows up here.
3. **FIFO restore holds under spec** — a younger session's verify must not
   overtake the resumed one. Read the restore order directly, do not infer it.
4. **Accounting:** `kv_committed_len` advances only through the deferred
   result processor across the spill/resume boundary; no second writer appears.
5. **The tick variant separately:** with `--kv-session-offload-spec-in-tick`,
   host-KV stream passes per accepted token fall by roughly the accept-length
   factor. That is the stated win and it is measurable.

Expected failures that are NOT bugs: PS2 deep prefill spill refused by name
when resume-under-spec is on; DFLASH not used for spilled sessions.

## 7 — What this note does not claim

It does not verify the resume path's internals — only that it exists, is
gated, and is documented as unobserved. It measures nothing. And §3's open
question (FIFO order once a resumed session is back in a live spec batch)
stands unanswered by design rather than by omission.
