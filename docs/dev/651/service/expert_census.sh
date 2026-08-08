#!/bin/bash
# #651 9.7: does expert routing on this checkpoint have a COLD TAIL?
#
# The MoE disk-spill idea lives or dies on this. If a stable minority of
# experts is rarely routed to, that minority can live on NVMe and the per-token
# miss cost stays small. If the distribution is flat, no spill size is usable,
# because every token would touch a disk-resident expert.
#
# Workload shape is deliberate: SHORT prompts, MANY generated tokens. Decode is
# what this GPU survives; a long prompt is sustained prefill and wedges it, and
# a wedge mid-census would cost the measurement and the GPU both.
set -u
BASE=http://127.0.0.1:31651

curl -s -m 10 -X POST "$BASE/start_expert_distribution_record" >/dev/null
echo "recording started"

i=0
for p in "Count from one to forty in words." \
         "Write a haiku about winter, then another about summer." \
         "List ten common Python built-in functions with one-line descriptions." \
         "Explain recursion in three short sentences." \
         "Name eight European capitals, one per line."; do
  i=$((i+1))
  curl -s -m 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen36-35b-a3b\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],
         \"chat_template_kwargs\":{\"enable_thinking\":false},
         \"temperature\":0,\"max_tokens\":220}" \
    | python3 -c "import json,sys
try:
    d=json.load(sys.stdin); u=d.get('usage',{})
    print('  req $i ok, completion_tokens=', u.get('completion_tokens'))
except Exception as e:
    print('  req $i FAILED', e)"
done

curl -s -m 10 -X POST "$BASE/stop_expert_distribution_record" >/dev/null
echo "recording stopped"
curl -s -m 60 -X POST "$BASE/dump_expert_distribution_record" > /root/651-p2/results/expert_dump.json 2>&1
echo "dump bytes: $(wc -c < /root/651-p2/results/expert_dump.json)"
head -c 400 /root/651-p2/results/expert_dump.json
