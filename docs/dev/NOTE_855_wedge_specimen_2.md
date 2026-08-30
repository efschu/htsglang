=== #855 WEDGE SPECIMEN 2 (gdncovB2, 2026-08-30 05:42:48Z onset)
log: /spinning/evidence-665-f1/boot_855_gdncovB2_0840f82601_0830_053228.log:53753ff
deadman FIRED 05:45:13Z (~2.5 min after onset); no process remained because the one-shot waiter EXITS when it fires -- working as designed, not a miss.

MECHANISM (the important part):
  PHASE-POLICY ARM-UNFUNDED: armed tp_to_pp (pending prefill 1 tok > 0
  (purity: prefill cannot run in tp, nothing decoding)) -- 12x FLIP ABANDONED.
  The 'pending prefill 1 tok' is the DEADMAN'S OWN /health_generate probe.
  So the liveness probe arms a flip that strict purity then forbids, and the
  probe-detected livelock is one the probe itself triggers. Matches the
  coordinator's #942 side-finding and fix/942-subchunk-no-arm @ 3b843bd4dd.

REPRODUCED ON BOTH CHECKPOINTS -- not the INT8 artifact:
  armA (incumbent vocabint8-embed) wedged 05:28Z: ADMISSION-WEDGE, 1 queued /
  0 running / no first token 438s, after a manual pp_to_tp flip.
  armB (gdncov union) wedged 05:05Z and 05:42Z on the flip/purity path.
