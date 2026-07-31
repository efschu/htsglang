# GPU phase of the BAR1 direct mode (#292)

The CPU phase is complete (`CUDA_VISIBLE_DEVICES=99` throughout, no card
touched). What follows is the command list for the cards -- executable
verbatim, in the order it has to run.

The rationale for each item lives in `docs/dev/INTEGRATION_R3_VALIDATION.md`,
section "BAR1 direct mode graph-safe (#292)". This sheet does not repeat it,
it executes it.

**The order is deliberate: byte proofs first, then numbers. No timing
measurement before a passed byte proof.**

---

## 0. Where this runs

The BAR1 work does **not** run where the battery runs. The patched driver,
`dmabuf_holder` and `/dev/dmabuf_holder` live on the PVE host; CT999 cannot
open the holder device (major 10 is not in the container's device
allowlist). So: commands over ssh on the host, artifacts in the container.

    Host            192.168.0.1
    Key             /root/.ssh/id_root@proxmox
    Path mapping    <container path>  ->  /spinning/subvol-999-disk-0<container path>

Every ssh call gets a **timeout**. An unbounded ssh in a single Bash call
makes the agent unreachable without anyone noticing
(`scripts/gpu_battery/battery_host.sh`, `host_ssh_for`).

### Preconditions on the host (04_BETRIEB.md of the P2P handover)

Check, don't assume:

    ssh -i /root/.ssh/id_root@proxmox root@192.168.0.1 \
      'grep -i "^RegistryDwords:" /proc/driver/nvidia/params; \
       ls -l /dev/dmabuf_holder; lsmod | grep -c "^dmabuf_holder"'

An empty `RegistryDwords` means the stock driver -- then direct mode does
not run at all, and any number from such a run measures something else.
Load per `04_BETRIEB.md`, "load driver" (`nvidia_modeset` must be unloaded
along with it, otherwise `rmmod nvidia` fails with "File exists").

### Environment for every host run

Verbatim from `scripts/gpu_battery/s11_bar1_e2e.sh`, just with this
branch's worktree:

    V=/spinning/subvol-999-disk-0/spinning/htsglang-gpu/.venv
    W=/spinning/subvol-999-disk-0/spinning/wt-direkt-graph
    N=/spinning/subvol-999-disk-0/spinning/nvidia-open-595
    P=/spinning/miniforge3_local_install/bin/python3.12

    cd $W
    PYTHONPATH=$W/python:$V/lib/python3.12/site-packages \
    LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/nvidia/cu13/lib \
    CUDA_HOME=$V/lib/python3.12/site-packages/nvidia/cu13 \
    SGLANG_BARLINK_BAR1_NV_SOURCE=$N \
    TORCH_EXTENSIONS_DIR=/spinning/subvol-999-disk-0/spinning/barlink_extcache_host \
    TORCH_CUDA_ARCH_LIST="8.6;12.0" MAX_JOBS=4 \
      $P <command>

`CUDA_HOME` is mandatory, otherwise the JIT build fails at `ninja`.
`MAX_JOBS=4`, because this box has no swap.

---

## 1. Locks

Two `/tmp` namespaces, and they cannot see each other: `/tmp/gpu-card-N.lock`
in CT999 and the identically named path on the host are **different**
locks. Whoever touches the cards takes **both**.

    # Container side (one per card)
    for i in 0 1 2; do mkdir /tmp/gpu-card-$i.lock || echo "TAKEN: $i"; done
    printf 'holder=direktmodus_gpu_phase\nstep=292\nacquired=%s\nheartbeat=%s\n' \
      "$(date -Is)" "$(date -Is)" | tee /tmp/gpu-card-{0,1,2}.lock/info

    # Host side, identical, over ssh (see host_locks_acquire)

`mkdir` is the atomic acquisition, the `info` file carries the holder and
heartbeat. **A foreign lock is never broken** -- read the holder from
`info`, ask the operator, abort. Refresh the heartbeat every 60 s, otherwise
the reaper (`/spinning/gpu-arb/arb-reaper.sh`) clears the lock out from
under the running attempt.

Release on **every** exit, including the error path.

---

## 2. The graph proof -- nine gate cases

    <environment from 0> benchmark/bar1_graph_check.py 0,1,2

Expected: **"All gate cases passed."** Nine gate cases appear in the
summary, plus `grid` as an info case (not a gate):

    [Gate]  1blk-small            [Gate]  pipe
    [Gate]  1blk-large            [Gate]  pipe-direct              <- new (#292)
    [Info]  grid                  [Gate]  pipe-direct-pool-empty   <- new (#292)
    [Gate]  reservation           [Gate]  broadcast
    [Gate]  two-graphs            [Gate]  broadcast-two-graphs

Just the two new ones:

    <environment> benchmark/bar1_graph_check.py 0,1,2 29593 pipe-direct,pipe-direct-pool-empty

`pipe-direct` checks two things no other case checks:

* the result tensor must **really sit in the BAR1 window** (`result_window()`).
  If that is missing, the case measured and passed the `direct=0` control
  path without answering the question. The case then aborts with a pointer
  to `SGLANG_BARLINK_BAR1_PIPE_RESULT_RING` instead of passing green;
* it is also read back over the **host** instead of over the receiving
  card's L2. L2 is not coherent with incoming PCIe writes
  (`BEFUND_L2_NICHT_KOHAERENT.md`), so the second read path here is not a
  formality (measurement discipline, rule 3).

`pipe-direct-pool-empty` is the negative control: with `ERG_RING=2` there
are no graph slots, **every** captured call must fall back to `direct=0`
and still deliver the right bytes. A case that finds a slot in the window
here reports a broken ring split.

---

## 3. Eager byte proof, with the handshake switched on

Before the graph, because a failed eager proof invalidates every graph
number.

    <environment> env \
      SGLANG_BARLINK_BAR1_PIPE=1 \
      SGLANG_BARLINK_BAR1_PIPE_DIRECT=1 \
      SGLANG_BARLINK_BAR1_PIPE_DIRECT_GRAPH=1 \
      SGLANG_BARLINK_BAR1_PIPE_RESULT_RING=5 \
      $P benchmark/bar1_diag.py 0,1,2

This makes the release handshake run eager too (`ergSlack = 2`), and flag
family 4 gets exercised on real hardware for the first time. Up to this
point it has only seen a Python simulation.

After **every** run, the cap:

    grep -c "Zeitlimit" <log>      # expected: 0

The handshake is a new wait condition. A tripped time cap there is the
first suspect whenever something hangs, and it invalidates every number
from that run.

---

## 4. Register count on sm_120, on the JIT object

Measured offline (`scripts/probe/bar1_pipe_spill.sh`), the grid variant on
sm_120 rises from REG 40 to 48, STACK stays 0. Whether that measurably
lowers occupancy is for the card to decide. Measured on the **built**
object, not the one compiled offline:

    ssh -i /root/.ssh/id_root@proxmox root@192.168.0.1 \
      'ls -l /spinning/barlink_extcache_host/barlink_bar1_pipe_ext_cuda_86_120/'

    ssh ... '/usr/local/cuda/bin/cuobjdump -res-usage \
      /spinning/barlink_extcache_host/barlink_bar1_pipe_ext_cuda_86_120/barlink_bar1_pipe_ext_cuda_86_120.so \
      | grep -E "Function|REG|STACK"'

**Read the timestamps first.** The extension cache is shared across boots
(a cold build costs minutes). If the `.so` is older than
`python/sglang/srt/distributed/device_communicators/barlink_bar1_pipe_ext.py`,
`cuobjdump` measures the **old** kernel -- the number is then worthless and
the directory has to go before anything else runs.

---

## 5. A/B: `DIRECT=0` against `DIRECT=1`

Direct mode is, to this day, compiled and **unmeasured**.

Rules for this measurement, all three non-negotiable:

* **Warmup**, otherwise the number is wrong (P-state ramp: 726 us without,
  95 us with). Drive to the working point beforehand, don't grow into it.
* Measure **interleaved**, not arm by arm: A,B,A,B,... in the same process,
  same sizes, same order. Two separate runs would also measure the clock
  and temperature state.
* **Noise floor first**: an A-against-A pass before A is reported against
  B. Whatever lies below the noise floor is not reported.

    <environment> env SGLANG_BARLINK_BAR1_PIPE=1 SGLANG_BARLINK_BAR1_PIPE_DIRECT=0 \
      $P benchmark/bar1_diag.py 0,1,2
    <environment> env SGLANG_BARLINK_BAR1_PIPE=1 SGLANG_BARLINK_BAR1_PIPE_DIRECT=1 \
      $P benchmark/bar1_diag.py 0,1,2

And the same for `SGLANG_BARLINK_BAR1_PIPE_DIRECT_GRAPH=0/1` at
`DIRECT=1` -- that is the question about the 8 registers from item 4.

A saved VRAM pass at the receiver is expected. Against this rig's PCIe
bottleneck that is small; **a null result is a possible and reportable
outcome**, not a failure. Report in ms/round, not tok/s.

---

## 6. Standard e2e run

With `SGLANG_BARLINK_GRAPH_ENABLE=1` and graph-safe direct mode, otherwise as
in `scripts/gpu_battery/s11_bar1_e2e.sh`.

Expect the graph pool to run low quickly on a real model (many call sites
per graph, many graphs), with the rest running `direct=0`. The notice for
this appears once per rank and names the numbers. That is the feature's
honest scope, not its failure: graph-safe direct mode carries a BOUNDED
number of captured call sites, and every slot costs
`roundup(max_bytes, 4096)` bytes in the BAR1 window.

---

## Abort criteria

Abort, release locks, report the finding -- do not proceed and do not
reconfigure to work around the obstacle. Changing a configuration to work
around an obstacle swaps out the question.

1. **A foreign lock** on one of the three cards (container or host side).
   Report the holder from `info`, ask the operator.
2. **`RegistryDwords` empty** or `/dev/dmabuf_holder` missing -- stock
   driver, direct mode does not run.
3. **A gate case from item 2 fails.** `SGLANG_BARLINK_GRAPH_ENABLE` stays
   off, items 5 and 6 do not run. The info case `grid` is allowed to fail.
4. **`pipe-direct` reports "result tensor NOT in the BAR1 window".** Then
   the run measured the control path; the resulting number says nothing
   about direct mode.
5. **Time cap tripped** (`Zeitlimit` in the log, `barlink.status()` nonzero).
   Every number from that run is invalidated.
6. **JIT object older than the source** (item 4). Cache gone, rebuild,
   start over.
7. **VRAM corridor violated**: free < 400 MiB absolute, or > 1.5 GiB net
   wasted. Do not measure into the red.
8. **A hang.** `py-spy dump` BEFORE every kill, and kill only your own
   PIDs -- never a broad `pkill`, other servers run on this box.
