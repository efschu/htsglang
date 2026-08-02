# Incident: SIGUSR2 killed the arm-A pilot boot (2026-08-02)

**What happened.** To flush a #390 expert-stats dump mid-run I sent SIGUSR2 to
every PID that matched the server. One of them was the FRONTEND process
(`sglang.launch_server`, PID 2489497). It has no MoE layers, so it never
constructs an `ExpertStatsCollector`, so `_install_dump_triggers_once()` never
ran there, so no SIGUSR2 handler was installed. **The default disposition of
SIGUSR2 is terminate.** The frontend died and took the scheduler ranks with it
before any dump was flushed.

**Cost.** One boot (~7 min load). The arm-A decode numbers had already been
taken and survive as `floor_A{1,2}_pilot.json`; the arm-A stats dump was lost.

**Root cause, stated precisely.** The #390 signal trigger is installed
PER-PROCESS and only in processes that build a collector. "The server" is not
one process. Broadcasting a signal whose default action is fatal across a
process group, on the assumption that a handler exists everywhere, is the
defect -- not the signal choice itself.

**Fix adopted.** `SGLANG_EXPERT_STATS_INTERVAL_SEC=45`. The interval dump needs
no signal at all, so there is nothing to aim at the wrong process. Both measured
boots use it.

**Standing rule for the runbook.** Never broadcast SIGUSR2 (or any signal with
a fatal default) to a matched process set. Either target the single PID that is
known to have installed the handler, or use a signal-free trigger. `py-spy dump`
first if the intent is diagnosis rather than a dump.
