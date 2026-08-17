# ANALYSE 741 — the barlink-BAR1 fault family: ONE defect, two presentations

Desk root-cause for the family isolated in `ANALYSE_734_sendbytes_specimens.md`
(specimens S2 and S3). Verdict and fix shape; the fix itself is desk-buildable
per the operator's ack, and any boot goes on F4-r4's window list.

    S2  2026-08-17 19:30:40  rank0 CUDA illegal memory access
                             barlink_bar1 poll_status_word
    S3  2026-08-05 20:59:38  Bar1CollectiveAborted, "a spin kernel took
                             its abort path", at all_reduce


## 1. Verdict: ONE defect

**`Bar1Transport.close()` releases the device memory the watchdog is still
polling, and disarms the watchdog fourteen lines later.**

`python/sglang/srt/distributed/device_communicators/barlink_bar1.py`, in
`close()` (`:5563`):

    :5643            self._cuda.vmm_free(*w)        # <-- memory RELEASED
    :5646        self._ctl_dev = None
                 ...
    :5657        self._abort_poll_active = False    # <-- watchdog disarmed

The intent is stated correctly in the code's own comment two lines above the
disarm:

> `# #616f: the watchdog poll holds a pinned page and a stream, and it`
> `# reads `_ctl_dev` directly -- it has to stand down with the word.`

It does not stand down *with* the word. It stands down *after* it. The
watchdog runs on its own thread (`poll_status_word`, `:4807`, "Runs on the
WATCHDOG thread, never on the serving path"), takes no lock, and is not joined
before the free. Any poll that lands in that window reads memory `vmm_free`
has already released.

**Both specimens are that read, and which one you get depends only on what
the allocator did with the block:**

| block state after free | what the poll reads | presentation |
|---|---|---|
| unmapped | fault | **CUDA illegal memory access** — S2 |
| recycled, still mapped | whatever now lives there | **non-zero garbage word** -> `check_aborted` raises `Bar1CollectiveAborted` — S3 |

The second arm needs one more fact to close, and it holds: **the abort code is
never validated.** `check_aborted` raises on "a kernel took its abort path"
for any non-zero status word; there is no known-code table anywhere in the
module (searched: no `ABORT_CODE` constants, no membership test). So a garbage
word is indistinguishable from a genuine trip — which is exactly why S3 reports
a spin-kernel abort that no spin kernel performed.

That is the unification. One root, two faces, and the face is chosen by the
CUDA caching allocator rather than by anything in the failure.


## 2. Why the log ordering does not incriminate the poll

`ANALYSE_734` warned that S2's ordering — abort gate first, payload path second
— cannot establish the gate as the origin. That is now confirmed by mechanism
rather than by caution.

`barlink_abort_gate.poll_status_words()` (`:327`) wraps each transport poll in

    except Exception:  # pragma: no cover - a watchdog must not die
        logger.exception("barlink-BAR1 status poll failed")

A CUDA illegal memory access is **sticky**: it poisons the context, so every
subsequent CUDA call in the process re-raises it. The gate swallows it and
keeps polling, so the same fault is re-reported on each pass — which is
precisely the "four repeated" errors in S2 — and the serving path only learns
about it when it next touches CUDA, which happened to be
`phase_flip_runtime.py:6926 _execute` -> `kv_reshard.py:359 _checksum` ->
`weights_arena.py:127 uint8_checksum`.

So the gate is the **first reporter**, not the perpetrator, and the payload
path is the **second reporter**, not the victim of a separate bug. This is the
same reporter-vs-perpetrator shape recorded for #722; here it recurs one layer
down, and the swallow is what makes it look like two faults.

The swallow is itself a defect for this class: `except Exception` is right for
a transient watchdog error and wrong for a sticky context kill, where
continuing produces noise and delays the real report.


## 3. Contributing conditions, each independently sufficient to keep the bug alive

1. **Check-then-use across threads.** `poll_status_word` guards with
   `if not self._abort_poll_active or self._ctl_dev is None:` and then uses
   both. Nothing makes the check and the use atomic against `close()`.
2. **No `record_stream()` anywhere in the module** (searched: zero hits). The
   poll issues `self._abort_poll_dst.copy_(self._ctl_dev[0:1],
   non_blocking=True)` on a PRIVATE stream (`self._abort_poll_stream`) against
   a tensor allocated on another stream. That is the documented PyTorch
   caching-allocator hazard on its own: the block may be recycled once the
   owning stream is done while the private stream's copy is still in flight —
   no teardown required.
3. **Nothing pauses the poll for a flip or a reshard.** `pause_polling()`
   exists and has exactly ONE real caller, `parallel_state.py:2888`, and its
   docstring says why it is there: CUDA graph capture, because "a synchronizing
   CUDA call in ANY thread of the process invalidates the capture". Teardown
   and arena remap were never given the same treatment, though they have the
   same cross-thread hazard.
4. **No abort-code validation** (section 1).

Condition 2 matters for the verdict's reach: it means the family does not
strictly require a `close()` to fire. A plain allocator recycle under memory
churn can do it. That is consistent with S3 occurring on a configuration with
no phase flip at all.


## 4. Why S2 landed on the phase-flip path specifically

S2's `server_args` reads `tp_size=1, pp_size=3`, with `enable_phase_flip=True`
and `phase_flip_tp_vector='32,16,16'`. With `tp_size=1` there is no
steady-state TP collective for barlink to serve — the BAR1 transport exists in
that boot **for the flip's TP layout**. So the flip is the thing that builds
and tears these transports down while the server is live, which is why a
teardown-ordering bug surfaces there and why S2's second reporter is on the
flip's own payload path.

S3 needs no flip: `tp_size=3, pp_size=1`, pure TP on a different model
(Qwen3.6-27B-INT8-W8A8). Same root, reached by the allocator rather than by a
`close()` — see condition 2.

That the two specimens differ in model, in parallelism, and in whether a phase
flip exists at all, while sharing one mechanism, is what makes this a family
rather than two incidents.


## 5. Fix shape

Ordered by what each buys; 5.1 is the root, the rest close the conditions that
would let it come back in another form.

**5.1 Disarm before free — the ordering the comment already intends.** In
`close()`, move `self._abort_poll_active = False` (and the stream/dst drops) to
BEFORE the `vmm_free` loop and the `_ctl_dev = None`. Disarming is not enough
on its own, because the watchdog may already be inside the poll: the disarm has
to be followed by a quiesce (join the watchdog, or take the same lock the poll
takes) before the free.

**5.2 Make the guard atomic.** A lock (or a generation counter the poll
re-checks after its copy) so that "still armed" cannot go stale between the
check and the read.

**5.3 `record_stream()` on `_ctl_dev` and `_round_dev` for
`_abort_poll_stream`**, or allocate the poll's source on the poll stream. This
closes the no-teardown variant and is the only item that addresses S3's
configuration directly.

**5.4 Pause polling around flip/reshard teardown**, reusing the existing
`pause_polling()` context manager rather than inventing a second mechanism.

**5.5 Narrow the gate's `except Exception`** so a sticky CUDA error is
re-raised (or latched and reported once) instead of swallowed and re-hit every
pass. This does not fix the fault; it stops the fault from being reported four
times in the wrong place.

**5.6 Validate the abort code.** Any non-zero currently means "a kernel took
its abort path". A known-code check turns a garbage read into a NAMED
"implausible abort word" refusal instead of a false accusation against a spin
kernel — i.e. it converts S3's presentation from misleading to diagnostic.

5.1 + 5.2 are the fix. 5.3 is required for the S3 arm. 5.5 + 5.6 are
diagnostics that would have made this root visible in one boot instead of
across twelve days.


## 6. Open

- **Lineage.** Which commit each specimen ran, #634's exact closure text and
  reopen trigger, and the #717/#722 dates are being pulled separately. The
  reopen decision for #634 depends on those dates and is NOT made here.
- **Whether `close()` is the flip's actual teardown entry** for S2. The
  ordering defect stands on its own reading, but I have not traced the flip's
  call into `close()` line by line; condition 2 means the family does not
  depend on that link, and the S2 attribution does.
- **No boot.** Everything above is read from source and from the two logs.
