#!/bin/bash
# #651: one real coding round trip through omp, AS efeu, against the local
# on-demand endpoint.
#
# Run as root; it drops to efeu itself, because the point of the test is that
# the configuration works for the USER who will use it -- an agent that only
# works as root is not installed, it is merely present. It writes into a
# scratch directory owned by efeu for the same reason.
#
# Non-interactive (-p) so it can be asserted on. If the model is parked this
# will take about two and a half minutes before a token appears.
set -u

OUT=${OUT:-/tmp/omp_smoke_$(date +%H%M%S).txt}

su - efeu -c '
  set -u
  export PATH=$HOME/.local/bin:$PATH
  WORK=$(mktemp -d /home/efeu/omp-smoke-XXXX)
  cd "$WORK"
  # A task with a checkable answer: the agent has to actually write the file,
  # not just talk about it, so the assertion below tests the tool path and not
  # only the chat path.
  timeout 900 omp --model local/qwen36-35b-a3b --no-lsp -p \
    "Create a file named reverse.py containing a Python function reverse(s) that returns the reversed string. Then stop." \
    2>&1 | tail -25
  echo "--- files produced ---"
  ls -la "$WORK"
  if [ -f "$WORK/reverse.py" ]; then
    echo "--- reverse.py ---"
    cat "$WORK/reverse.py"
    echo "SMOKE: file written"
  else
    echo "SMOKE: no file written"
  fi
' 2>&1 | tee "$OUT"

echo "smoke output: $OUT"
