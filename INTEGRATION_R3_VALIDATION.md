
## #107 follow-up 2 addendum — output validation (re-boots, same recipe)

Response bodies of the measurement runs were not retained, so one re-boot per
arm with the identical recipe and prompt, 2400 tokens each, full text saved
(/tmp/q3trade/text_arm{S,B}.json). Cross-check that the re-boots measure the
same operating point: accept 2.550/2.510 vs 2.609/2.534, tok/s 96.5/75.0 vs
98.5/74.7 — both arms reproduce within ~2 %.

Verdict both arms: coherent. Structured technical prose exactly on the
prompt's five topics, no repetition pathology (TTR 0.326/0.418, top bigrams
are domain terms: "kv cache" 18/21, "chunked prefill" 8/10). The visible
difference (arm S answered directly, arm B stayed in the think block) is the
documented content-mode bimodality at temperature 0.7 — the modes were
swapped between arms in the original runs, so it is not an arm effect.

Process note: pgrep -f "sglang.launch_server" hit the agent's own shell
during teardown (documented self-kill trap, second occurrence today); fixed
by separating kill and boot into distinct calls and filtering own PID.
